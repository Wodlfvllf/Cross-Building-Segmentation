from model import get_model
from util import *
from tqdm import tqdm

def reduce_fn(vals):
    # take average
    return sum(vals) / len(vals)
def train_one_epoch(epoch, epochs, model, train_dataloader, valid_dataloader, criterion, scheduler, device, optimizer):
    train_loss = []
    train_iou = []
    valid_loss = []
    valid_iou = []

    model.train()
    scheduler.step()
    for step, batch in tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch+1}/{epochs} (Train)', unit='batch'):
        image = batch['img'].to(device)
        mask = batch['mask'].to(device)
                               
        optimizer.zero_grad()

        outputs = model(image)
        loss, iou_score = criterion(outputs, mask)
        loss.sum().backward()
        
        xm.optimizer_step(optimizer)
        loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn) 
        iou_score_reduced = xm.mesh_reduce('iou_reduce',iou_score,reduce_fn)
        
        if step%100 == 0:
            xm.master_print(f'Train_Batch: {step}, loss: {loss_reduced}  iou_score: {iou_score_reduced}')

        train_loss.append(loss_reduced.detach().cpu().numpy())
        train_iou.append(iou_score_reduced.detach().cpu().numpy())
        gc.collect()

    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(valid_dataloader), desc=f'Epoch {epoch+1}/{epochs} (Train)', unit='batch'):
            image = batch['img'].to(device)
            mask = batch['mask'].to(device)
            outputs = model(image)

            loss, iou_score = criterion(outputs, mask)
            loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn) 
            iou_score_reduced = xm.mesh_reduce('iou_reduce',iou_score,reduce_fn)
            
            if step%100 == 0:
                xm.master_print(f'Train_Batch: {step}, loss: {loss_reduced}  iou_score: {iou_score_reduced}')
                gc.collect

            valid_loss.append(loss_reduced.detach().cpu().numpy())
            valid_iou.append(iou_score_reduced.detach().cpu().numpy())
            
    if epoch % 2 == 0:
        visualize(
            original_image = image[0].permute(1, 2, 0).detach().cpu().numpy(),
            ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask[0].permute(1, 2, 0).detach().cpu().numpy()), select_class_rgb_values),
            output_mask = reverse_one_hot(torch.sigmoid(outputs[0]).permute(1, 2, 0).detach().cpu().numpy())
        )
    return np.mean(train_loss), np.mean(valid_loss), np.mean(train_iou), np.mean(valid_iou)

def data():
    class_names = ['background', 'building']
    select_classes = ['background', 'building']
    class_rgb_values = [[0], [1]]
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]


    train_transform = A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512, p=1),
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p = 0.5),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512, p=1),
        ToTensorV2()
    ])
    
    ##Train Dataset
    train_dataset = Custom_Dataset('/home/shashank/IEEE/train/', '/home/shashank/IEEE/train/train.json', transforms = transform, 
                                   class_rgb_values=select_class_rgb_values)

    ##Test Dataset
    test_dataset = Custom_Dataset('/home/shashank/IEEE/val/', '/home/shashank/IEEE/val/val.json', transforms = val_transform, 
                                  class_rgb_values=select_class_rgb_values)
    
    ##Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                                    train_dataset,
                                                                    num_replicas=xm.xrt_world_size(),
                                                                    rank=xm.get_ordinal(),
                                                                    shuffle = True
                                                                    )
    
    ##Train Dataloader
    dataloader_train = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=16,
                                                    sampler = train_sampler,
                                                    drop_last = True,
                                                    num_workers=4,
                                                    pin_memory = True)
    
    
    ##Sampler
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                                                                    test_dataset,
                                                                    num_replicas=xm.xrt_world_size(),
                                                                    rank=xm.get_ordinal(),
                                                                    shuffle = False
                                                                    )
    ##Test Dataloader
    dataloader_test = torch.utils.data.DataLoader(test_dataset,
                                                  sampler = valid_sampler,
                                                  batch_size=16,
                                                  drop_last = True,
                                                  num_workers=4,
                                                 )
    return dataloader_train, dataloader_test, train_dataset.__len__()


def train_function(model, length, epochs):
    criterion = custom_loss
    lr = 0.00001
    num_train_steps = int(
        length / 16 / xm.xrt_world_size() * epochs
    )

    lr = lr * xm.xrt_world_size()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    train_dataloader, test_dataloader, length = data()
    
    xm.master_print(f'num_training_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')
    device = xm.xla_device()
    model = model.to(device)
    best_iou = 0.0
    train_loss = []
    valid_loss = []
    for epoch in (range(epochs)):
        gc.collect()
        para_loader = pl.ParallelLoader(train_dataloader, [device])
        train_loader = para_loader.per_device_loader(device)

        val_loader = pl.ParallelLoader(test_dataloader, [device])
        valid_loader = val_loader.per_device_loader(device)
        
        trn_loss, val_loss, train_iou, valid_iou = train_one_epoch(epoch, epochs, model, train_loader, valid_loader, criterion, scheduler, device, optimizer)
        
        scheduler.step()
        train_loss.append(trn_loss)
        valid_loss.append(val_loss)
        gc.collect()

        if best_iou < valid_iou:
            best_iou = valid_iou
            xm.save(model.state_dict(), f'./unet_model.pth')
        else:
            model.load_state_dict(torch.load('./unet_model.pth'))
        
        xm.master_print(f'Epoch {epoch+1}/{epochs}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}, Train IOU: {train_iou}, Valid IOU: {valid_iou}')
        with open('losses.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{epochs}, Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f},  Train IOU: {train_iou}, Valid IOU: {valid_iou}\n')
        
    xm.rendezvous('save_model')
    xm.master_print('save model')
    xm.save(model.state_dict(), f'./unet_model.pth')


def _mp_fn(rank, flags):
    try:
        model = get_model()
        dev = xm.xla_device()
        model = model.to(dev)
        dataloader_train, dataloader_test, length = data()
        torch.set_default_tensor_type('torch.FloatTensor')
        train_function(model, length, epochs=30)
        # xser.save(model.state_dict(), f"model.bin", master_only=True)
    except Exception as e:
        print(f"Exception in process {rank}: {e}")
        raise

if __name__ == '__main__':
    FLAGS = {}
    try:
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=4, start_method='fork')
    except Exception as e:
        print(f"Exception in main: {e}")