import tempfile
from datasets import load_dataset
import tiktoken
from functools import partial
from helper_function import *
from torch.utils.data import Dataset, DataLoader
from models_1 import *
from load_gpt2 import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from tqdm import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group()

def main(rank, world_size, n_epochs, eval_step, data_path, ckpt_path, save_every, b_sz, load_from, lr):
    setup(rank, world_size)
    train_set, val_set, test_set = prepare_dataset(data_path)
    torch.distributed.barrier()
    customized_collate_fn = partial(
        custom_collate_fn,
        device = 'cpu',
        allowed_max_length=1024
    )

    train_loader = DataLoader(
        train_set,
        batch_size=b_sz,
        collate_fn=customized_collate_fn,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_set),
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=b_sz,
        collate_fn=customized_collate_fn,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(val_set),
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=b_sz,
        collate_fn=customized_collate_fn,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_set),
        drop_last=True
    )
    
    print('data ready')
    # data ready=================
    model = prepare_model(load_from)
    tokenizer = tiktoken.get_encoding("gpt2") #set up tokenizer
    # model ready================
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=n_epochs)
    
    trainer = Trainer(model, tokenizer, train_loader, val_loader, test_loader, optimizer, scheduler, rank, ckpt_path, save_every)
    if rank == 0:
        print('trainer ready, start training')
    torch.distributed.barrier()
    trainer.train(n_epochs, eval_step)
    torch.distributed.barrier()
    if rank == 0:
        trainer.save(ckpt_path)
    cleanup()

def prepare_dataset(file_path):
    with open(file_path, "r") as file:
        inst_data = json.load(file)
    # devide data into training testing validating
    train_portion = int(len(inst_data) * 0.85)  # 85% for training
    test_portion = int(len(inst_data) * 0.1)    # 10% for testing
    val_portion = len(inst_data) - train_portion - test_portion  # Remaining 5% for validation
    
    train_data = inst_data[:train_portion]
    test_data = inst_data[train_portion:train_portion + test_portion]
    val_data = inst_data[train_portion + test_portion:]
    
    tokenizer = tiktoken.get_encoding("gpt2") #set up tokenizer
    
    # set up datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    return train_dataset, val_dataset, test_dataset
    
def prepare_model(load_from = None):
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    CHOOSE_MODEL = "gpt2-medium (355M)"
    
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, 
        models_dir="gpt2"
    )
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    if load_from != None:
        model.load_state_dict(torch.load(load_from))
    return model


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gpu_id: int,
        ckpt_path,
        save_every
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = DDP(model, device_ids=[gpu_id])
        self.ckpt_path = ckpt_path
        self.save_every = save_every

    def _run_batch(self, inputs, targets, eval=False):
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if eval == False:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            return loss.item()
        return loss.item()

    def _run_epoch(self, epoch, eval_step):
        self.train_loader.sampler.set_epoch(epoch)
        step = 0
        total_loss = 0
        self.model.train()
        print(len(self.train_loader))
        # set the time bar for training monitor
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}", disable=self.gpu_id != 0)
        for step, (input_batch, target_batch) in enumerate(self.train_loader, 1):
            # send batch to correct gpu
            input_batch = input_batch.to(self.gpu_id)
            target_batch = target_batch.to(self.gpu_id)
            # calculate batch loss
            loss = self._run_batch(input_batch, target_batch)
            total_loss += loss
            # step and pbar update
            pbar.update(1)
            step += 1
            # evaluation loop
            if step % eval_step == 0 and self.gpu_id == 0 and step != len(self.train_loader):
                avg_train_loss = total_loss / eval_step
                val_loss = self._validate()
                #print(f'Epoch {epoch}, Step {step}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
                pbar.write(f'Epoch {epoch}, Step {step}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
                pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'}, refresh=True)
                total_loss = 0
                #torch.cuda.synchronize()  # Ensure all processes are synced
                self.model.train()
            # save model checkpoint loop
            if self.gpu_id == 0 and step % self.save_every == 0:
                self.save(path = self.ckpt_path)
        pbar.close()
                
    def _validate(self):
        self.model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for input_batch, target_batch in self.val_loader:
                input_batch = input_batch.to(self.gpu_id)
                target_batch = target_batch.to(self.gpu_id)
                loss = self._run_batch(input_batch, target_batch, eval=True)
                total_val_loss += loss
                num_val_batches += 1
                if num_val_batches >= 5:
                    break
        return total_val_loss / num_val_batches
        
    def train(self, max_epochs, eval_step):
        for epoch in range(max_epochs):
            self._run_epoch(epoch, eval_step)

    def save(self, path):
        torch.save(self.model.module.state_dict(), path)
        
def run(demo_fn, world_size, n_epochs, eval_step, data_path, ckpt_path, save_every, batch_size, load_from, lr):
    mp.spawn(main, args=(world_size, n_epochs, eval_step, data_path, ckpt_path, save_every, batch_size, load_from, lr), nprocs=world_size, join=True)
    print('train finished')

def test():
    print(1)