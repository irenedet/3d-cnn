import torch


# apply training for one epoch
def train(model, loader, optimizer, loss_function,
          epoch, device, log_interval=100, tb_logger=None):
    # set the model to train mode
    model.train()

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model, calculate loss and run backwards pass
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(x),
                len(loader.dataset),
                       100. * batch_id / len(loader), loss.item()))

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.log_scalar(tag='train_loss', value=loss.item(),
                                 step=step)
            #    # check if we log images in this iteration
            #    log_image_interval = tb_logger.log_image_interval
            #    if step % log_image_interval == 0:
            #        pshape = prediction.shape
            #        tb_logger.log_image(tag='input', image=crop_tensor(x, pshape)[0, 0].to('cpu'), step=step)
            #        tb_logger.log_image(tag='target', image=crop_tensor(y, pshape)[0, 0].to('cpu'), step=step)
            #        tb_logger.log_image(tag='prediction', image=prediction[0, 0].to('cpu').detach(), step=step)


# run validation after training epoch
def validate(model, loader, loss_function, metric, step=None, tb_logger=None):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():

        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y).item()
            val_metric += metric(prediction,
                                 crop_tensor(y, prediction.shape)).item()

    # normalize loss and metric
    val_loss /= len(loader.dataset)
    val_metric /= len(loader.dataset)

    if tb_logger is not None:
        assert step is not None, "Need to know the current step to log validation results"
        tb_logger.log_scalar(tag='val_loss', value=val_loss, step=step)
        tb_logger.log_scalar(tag='val_metric', value=val_metric, step=step)
        # we always log the last validation images
        # pshape = prediction.shape
        # tb_logger.log_image(tag='val_input', image=crop_tensor(x, pshape)[0, 0].to('cpu'), step=step)
        # tb_logger.log_image(tag='val_target', image=crop_tensor(y, pshape)[0, 0].to('cpu'), step=step)
        # tb_logger.log_image(tag='val_prediction', image=prediction[0, 0].to('cpu'), step=step)

    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(
        val_loss, val_metric))


    # build default-unet with sigmoid activation
    # to normalize prediction to [0, 1]
