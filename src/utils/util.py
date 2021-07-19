import torch
from transformers import get_linear_schedule_with_warmup


def save_model(model, output_path):
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_path)


def to_parallel(model):
    model = torch.nn.DataParallel(model)
    return model


def get_scheduler(batch_size, grad_acc, epochs, warmup_propotion, optimizer, len_train_data):
    num_train_steps = int(epochs * len_train_data / batch_size / grad_acc)
    num_warmup_steps = int(num_train_steps * warmup_propotion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_train_steps,
    )

    return scheduler

def decode_iob(preds, attributes):
    iobs = []
    idx2iob = ["O", "B", "I"]
    for attr_idx in range(len(attributes)):
        attr_iobs = preds[attr_idx]
        attr_iobs = [[idx2iob[idx] + "-" + attributes[attr_idx] if idx2iob[idx] != "O" else "O" for idx in iob] for iob in attr_iobs]

        iobs.extend(attr_iobs)

    return iobs
