import argparse
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.programs import TransformerProgramModel, gumbel_soft, argmax
from utils import logging, data_utils, code_utils

logger = logging.get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch") # Output directory
    
    # Data 
    parser.add_argument("--dataset", type=str, default="reverse")               # Dataset name (reverse, hist, double_hist, sort, most_freq, dyck1 & dyck2)
    parser.add_argument("--vocab_size", type=int, default=8)                    # Vocabulary size (8, 1 for dyck1 & 2 for dyck2)
    parser.add_argument("--dataset_size", type=int, default=-1)                 # dataset size (20.000)
    parser.add_argument("--train_min_length", type=int, default=1)              # Minimum length of the training vocabulary
    parser.add_argument("--train_max_length", type=int, default=8)              # Maximum length of the training vocabulary (8, or 16 for dyck1 & dyck2)
    parser.add_argument("--test_min_length", type=int, default=1)               # Minimum length of the test vocabulary
    parser.add_argument("--test_max_length", type=int, default=8)              # Maximum length of the test vocabulary (8, or 16 for dyck1 & dyck2)
    
    parser.add_argument("--min_length", type=int, default=1)                    # Minimum length of the vocabulary (default: 1)
    parser.add_argument("--max_length", type=int, default=8)                    # Maximum length of the vocabulary (8, or 16 for dyck1 & dyck2)
    parser.add_argument("--seed", type=int, default=random.randint(0, 100000))  # Seed for random number generator (five random seeds)
    parser.add_argument("--do_lower", type=int, default=0)                      # Lowercase the dataset (default: 0)
    parser.add_argument("--unique", type=int, default=1)                        # Unique dataset (default: 1)
    parser.add_argument("--replace_numbers", type=int, default=0)               # Replace numbers with <num> (default: 0)

    # Model
    parser.add_argument("--n_vars_cat", type=int, default=1)                    # Number of categorical variables (default: 1)
    parser.add_argument("--n_vars_num", type=int, default=1)                    # Number of numerical variables (default: 1)
    parser.add_argument("--d_var", type=int, default=None)                      # Dimension of the variable (max_length)
    parser.add_argument("--n_heads_cat", type=int, default=2)                   # Number of categorical heads (2 for hist, double_hist & dyck2, or 4 for reverse, sort, most_freq & dyck1)
    parser.add_argument("--n_heads_num", type=int, default=2)                   # Number of numerical heads (2 for hist, double_hist & dyck2, or 4 for reverse, sort, most_freq & dyck1)
    parser.add_argument("--d_mlp", type=int, default=64)                        # Dimension of the MLP (default: 64)
    parser.add_argument("--n_cat_mlps", type=int, default=1)                    # Number of categorical MLPs (1 for reverse, hist, double_hist & dyck1, or 2 for sort, most_freq & dyck2)
    parser.add_argument("--n_num_mlps", type=int, default=1)                    # Number of numerical MLPs (1 for reverse, hist, double_hist & dyck1, or 2 for sort, most_freq & dyck2)
    parser.add_argument("--mlp_vars_in", type=int, default=2)                   # MLP variables input (default: 2)
    parser.add_argument("--n_layers", type=int, default=1)                      # Number of layers (3, or 1 for hist)
    parser.add_argument("--sample_fn", type=str, default="gumbel_soft")         # Sampling function (default: gumbel_soft)
    parser.add_argument("--one_hot_embed", action="store_true")                 # One hot embedding (default: False)
    parser.add_argument("--count_only", action="store_true")                    # Count only (default: False)
    parser.add_argument("--selector_width", type=int, default=0)                # Selector width (default: 0)
    parser.add_argument("--attention_type", type=str, default="cat")            # Attention type (default: cat)
    parser.add_argument("--rel_pos_bias", type=str, default="fixed")            # Relative positional bias (default: fixed)
    parser.add_argument("--mlp_type", type=str, default="cat")                  # MLP type (default: cat)
    parser.add_argument("--autoregressive", action="store_true")                # Autoregressive (default: False)

    parser.add_argument(
        "--glove_embeddings", type=str, default="data/glove.840B.300d.txt"
    )                                                                           # Glove embeddings (data/glove.840B.300d.txt)
    parser.add_argument("--do_glove", type=int, default=0)                      # Do glove (default: 0)

    parser.add_argument("--unembed_mask", type=int, default=1)                  # Unembed mask (default: 1)
    parser.add_argument("--pool_outputs", type=int, default=0)                  # Pool outputs (default: 0)

    # Standard model
    parser.add_argument("--standard", action="store_true")                      # Standard model (default: False)
    parser.add_argument("--d_model", type=int, default=64)                      # Dimension of the model (default: 64)
    parser.add_argument("--d_head", type=int, default=None)                     # Dimension of the heads (default: None)
    parser.add_argument("--n_heads", type=int, default=2)                       # Number of heads (default: 2)
    parser.add_argument("--dropout", type=float, default=0.0)                   # Dropout rate (default: 0.0)

    # Training
    parser.add_argument("--lr", type=float, default=5e-2)                       # Learning rate (default: 5e-2)
    parser.add_argument("--max_grad_norm", type=float, default=None)            # Maximum gradient norm (default: None)
    parser.add_argument("--gumbel_samples", type=int, default=1)                # Gumbel samples (default: 1)
    parser.add_argument("--n_epochs", type=int, default=250)                    # Number of epochs (default: 250)
    parser.add_argument("--batch_size", type=int, default=512)                  # Batch size (default: 512)
    parser.add_argument("--tau_init", type=float, default=3.0)                  # Gumbel temperature initialization (default: 3.0)
    parser.add_argument("--tau_end", type=float, default=0.01)                  # Gumbel temperature end (default: 0.01)
    parser.add_argument("--tau_schedule", type=str, default="geomspace")        # Gumbel temperature schedule (default: geomspace)
    parser.add_argument("--loss_agg", type=str, default="per_token")            # Loss aggregation (default: per_token)

    parser.add_argument("--save", action="store_true")                          # Save model (default: False)
    parser.add_argument("--save_code", action="store_true")                     # Save code (default: False)

    parser.add_argument("--device", type=str, default="cuda")                   # Device (cuda, cpu, mps)
        
    args = parser.parse_args()
    
    logging.initialize(args.output_dir) # Log arguments to output.log
    
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_test(
    model,                      # 1. TModel (train), 2. TModel (validation), 3. TModel (test), 4. TModel (train), 5. TModel (validation), 6. TModel (test)
    X,                          # 1. Array (129600), 2. Array (14400), 3. Array (16000), 4. Array (129600), 5. Array (14400), 6. Array (16000)
    Y,                          # 1. Array (129600), 2. Array (14400), 3. Array (16000), 4. Array (129600), 5. Array (14400), 6. Array (16000)
    batch_size=256,             # 1. 512, 2. 512, 3. 512, 4. 256, 5. 256, 6. 256
    x_pad_idx=0,                # 1. 0, 2. 0, 3. 0, 4. 0, 5. 0, 6. 0
    y_pad_idx=0,                # 1. 0, 2. 0, 3. 0, 4. 0, 5. 0, 6. 0
    autoregressive=False,       # 1. False, 2. False, 3. False, 4. False, 5. False, 6. False
    func=torch.argmax,          # 1. argmax, 2. argmax, 3. argmax, 4. argmax, 5. argmax, 6. argmax
):
    dataloader = DataLoader(
        list(zip(X, Y)), batch_size=batch_size, shuffle=False
    )
    out = []
    preds = []
    true = []
    model.eval()
    for x, y in dataloader:
        x = x.to(model.device) # Define x as a tensor on the device (cuda, cpu, mps)
        m = (x != x_pad_idx).float() # Define mask m typically used to mask the padding tokens
        mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool() # Define 2D mask from 1D mask (mask[i, j] is True if both x[i] and x[j] are not padding elements and False otherwise)
        if autoregressive: # autoregressive = False (default) / = True (Dyck1 & Dyck2)
            mask = torch.tril(mask) # Lower triangular part of the matrix
        with torch.no_grad():
            log_probs = model(x, mask=mask).log_softmax(-1) # Calculate the log softmax of the logits to turn the logits into probabilities and the log function is applied for numerical stability
        tgts = y.to(model.device)
        all_losses = -log_probs.gather(2, tgts.unsqueeze(-1)).squeeze(-1) # Calculate the negative log likelihood of the target tokens
        masked_losses = all_losses.masked_fill((tgts == y_pad_idx), 0.0) # Mask the losses to ignore the padding tokens
        lengths = (tgts != y_pad_idx).sum(-1) # Calculate the length of the target tokens by counting the non-padding tokens
        losses = masked_losses.sum(-1) / lengths # Calculate the average loss per sequence
        out.append(losses.detach().cpu().numpy()) # Append the average loss per sequence to the out list
        pred = func(log_probs, -1) # Calculate the predictions by taking the argmax of the log probabilities
        preds.append(pred.detach().cpu().numpy()) # Append the predictions to the preds list
        true.append(tgts.detach().cpu().numpy()) # Append the targets to the true list
    preds = np.concatenate(preds, 0) # Concatenate the predictions
    true = np.concatenate(true, 0) # Concatenate the targets
    m = true != y_pad_idx # Define mask m that indicates which elements in the targets are not padding
    acc = (preds == true)[m].mean() # Calculate the accuracy by taking the mean of the correct predictions over the non-padding elements
    loss = np.concatenate(out, 0).mean() # Calculate the mean loss
    return loss, acc


def run_training(
    model,                      # 1. TModel
    opt,                        # 1. Adam
    X_train,                    # 1. Array (129600)
    Y_train,                    # 1. Array (129600)
    eval_splits=None,           # 1. [("val", X_val, Y_val), ("test", X_test, Y_test)]
    batch_size=256,             # 1. 256
    n_epochs=5,                 # 1. 250
    temps=None,                 # 1. np.geomspace(args.tau_init, args.tau_end, n_epochs)
    x_pad_idx=0,                # 1. 0
    y_pad_idx=0,                # 1. 0
    autoregressive=False,       # 1. False
    smooth_temps=True,          # 1. True
):
    train_dataloader = DataLoader(
        list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True
    )
    
    out = []
    metrics = []
    t = tqdm(range(n_epochs), total=n_epochs) # Initialze tqdm - progress bar library        
    for epoch in t:
        temp = temps[epoch]
        model.train()
        
        if temp is not None and smooth_temps and epoch + 1 < len(temps): # temp = np.geomspace value, smooth_temps = True (default) & epoch + 1 < epochs  -> True
            ttemps = np.geomspace(temp, temps[epoch + 1], len(train_dataloader))
        elif temp is not None: # temp = np.geomspace value -> True
            ttemps = [temp] * len(train_dataloader)
        else: # False
            ttemps = [None] * len(train_dataloader)
        
        epoch_losses = []
        for ttemp, (x, y) in zip(ttemps, train_dataloader):
            if ttemp is not None: # ttemp = np.geomspace value (line 165) -> True
                model.set_temp(ttemp)
            x = x.to(model.device) # Define x as a tensor on the device (cuda, cpu, mps)
            m = (x != x_pad_idx).float() # Define mask m typically used to mask the padding tokens
            mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool() # Define 2D mask from 1D mask (mask[i, j] is True if both x[i] and x[j] are not padding elements and False otherwise)
            if autoregressive: # autoregressive = False (default) / = True (Dyck1 & Dyck2)
                mask = torch.tril(mask) # Lower triangular part of the matrix
            lst = []
            losses_lst = []
            tgts = y.to(model.device) # Define tgts as a tensor on the device (cuda, cpu, mps)
            
            log_probs = model(x, mask=mask).log_softmax(-1) # Log softmax of the logits to get the logits into probabilities and the log function is applied for numerical stability
            all_losses = -log_probs.gather(2, tgts.unsqueeze(-1)).squeeze(-1) # Calculate the negative log likelihood of the target tokens
            masked_losses = all_losses.masked_fill((tgts == y_pad_idx), 0.0) # Mask the losses to ignore the padding tokens
            lengths = (tgts != y_pad_idx).sum(-1) # Calculate the length of the target tokens by counting the non-padding tokens
            losses = masked_losses.sum(-1) / lengths # Calculate the average loss per sequence
            
            loss = losses.mean() # Calculate the mean loss
            lst.append(loss)
            losses_lst.append(losses.detach().cpu())
            
            loss = torch.stack(lst, 0).mean(0) # Calculate the mean loss over the samples
            loss.backward() # Backward pass to calculate the gradients
            if torch.isnan(loss):
                m = torch.isnan(losses_lst[-1]) # Define mask m that indicates which elements in the losses_lst are nan
                # Print the loss, the indices of the nan elements, the log probabilities, the input and the targets
                print(losses_lst[-1])
                print(m.nonzero())
                print(log_probs[m])
                print(x[m], tgts[m])
                raise ValueError("loss is nan")
            
            epoch_losses.append(torch.stack(losses_lst, 0).mean(0).numpy()) # Append the mean loss for the epoch
            opt.step() # Update the parameters
            opt.zero_grad() # Zero the gradients to decontaminate the gradients
            model.zero_grad() # Zero the gradients to decontaminate the gradients
        epoch_loss = np.concatenate(epoch_losses, 0).mean() # Join the losses for the epoch and calculate the mean loss
        d = {"loss": epoch_loss.mean()}
        t.set_postfix(d) # Set the progress bar postfix to mean loss
        out.append(epoch_loss.mean()) # Append the mean loss to the out list
        model.eval() # Set the model to evaluation mode
        
        with torch.no_grad():
            d = {
                "epoch": epoch,
                "epoch_loss": epoch_loss.mean(),
                "split": "train",
            } # Dictionary with the current epoch, the mean loss and the train split
            
            d["loss"], d["acc"] = run_test(
                model=model,
                X=X_train,
                Y=Y_train,
                x_pad_idx=x_pad_idx,
                y_pad_idx=y_pad_idx,
                autoregressive=autoregressive,
            ) # Run the test on the training data - Line 275
            
            # d.update(m) # Update the dictionary with the metrics (precision, recall, f1 score)
            metrics.append(d) 
            
            for split, X, Y in eval_splits: # eval_splits = [("val", X_val, Y_val), ("test", X_test, Y_test)] (parameter)
                d = {
                    "epoch": epoch,
                    "epoch_loss": epoch_loss.mean(),
                    "split": split,
                } # Dictionary with the current epoch, the mean loss and the split (val or test)
                d["loss"], d["acc"] = run_test(
                    model=model,
                    X=X,
                    Y=Y,
                    batch_size=batch_size,
                    x_pad_idx=x_pad_idx,
                    y_pad_idx=y_pad_idx,
                    autoregressive=autoregressive,
                ) # Run the test on the validation or test data - Line 275
                
                # d.update(m) # Update the dictionary with the metrics (None)
                metrics.append(d) # Append the dictionary to the metrics list

    return pd.DataFrame(metrics)
        

def run_program(
    args,
    idx_w=None,
    w_idx=None,
    idx_t=None,
    t_idx=None,
    X_train=None,
    Y_train=None,
    X_test=None,
    Y_test=None,
    X_val=None,
    Y_val=None,
):
    d = args.d_var
    init_emb = None
    unembed_mask = np.array([t in ("<unk>", "<pad>") for t in idx_t]) # If the element in idx_t is either <unk> or <pad>, the corresponding element in unembed_mask is True, otherwise it's False.
    model = TransformerProgramModel(
        d_vocab=len(idx_w),
        d_vocab_out=len(idx_t),
        n_vars_cat=args.n_vars_cat,
        n_vars_num=args.n_vars_num,
        d_var=d,
        n_heads_cat=args.n_heads_cat,
        n_heads_num=args.n_heads_num,
        d_mlp=args.d_mlp,
        n_cat_mlps=args.n_cat_mlps,
        n_num_mlps=args.n_num_mlps,
        mlp_vars_in=args.mlp_vars_in,
        n_layers=args.n_layers,
        n_ctx=X_train.shape[1],
        sample_fn=gumbel_soft,
        init_emb=init_emb,
        attention_type=args.attention_type,
        rel_pos_bias=args.rel_pos_bias,
        unembed_mask=unembed_mask,
        pool_outputs=args.pool_outputs,
        one_hot_embed=args.one_hot_embed,
        count_only=args.count_only,
        selector_width=args.selector_width,
    ).to(torch.device(args.device))
    
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr) # Initialize Adam optimizer
    n_epochs = args.n_epochs
    set_seed(args.seed) # Set seed - Line X (Redundant!!??)
    out = run_training(
        model=model,
        opt=opt,
        X_train=X_train,
        Y_train=Y_train,
        eval_splits=[("val", X_val, Y_val), ("test", X_test, Y_test)],
        batch_size=args.batch_size,
        n_epochs=n_epochs,
        autoregressive=args.autoregressive,
        temps=np.geomspace(args.tau_init, args.tau_end, n_epochs),
        x_pad_idx=w_idx["<pad>"], # Define x_pad_idx as the index of the padding token in the input vocabulary
        y_pad_idx=t_idx["<pad>"], # Define y_pad_idx as the index of the padding token in the target vocabulary
    ) # Run training - Line X
    
    out["sample_fn"] = args.sample_fn # Define the sample function (Default: gumbel_soft)
    model.set_temp(args.tau_end, argmax) # Set the temperature to the end temperature and the sample function to argmax
    dfs = [out]
    
    for split, X, Y in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        loss, acc = run_test(
            model=model,
            X=X,
            Y=Y,
            x_pad_idx=w_idx["<pad>"], # Define x_pad_idx as the index of the padding token in the input vocabulary
            y_pad_idx=t_idx["<pad>"], # Define y_pad_idx as the index of the padding token in the target vocabulary
            autoregressive=args.autoregressive,
        ) # Run test - Line 275
        
        logger.info(f"end ({split}): loss={loss}, acc={acc}") # Log the loss, accuracy and metrics for the split
        
        df = pd.DataFrame(
            {
                "epoch": [n_epochs],
                "split": split,
                "loss": loss,
                "acc": acc,
                "sample_fn": "argmax",
            }
        )
        dfs.append(df) # Append the dataframe to the dfs list (dfs is [{epoch, epoch_loss, split, loss, accuracy, }])
    df = pd.concat(dfs).reset_index(drop=True) # Concatenate the dataframes and reset the indices
    
    if args.save: # save = True -> True
        fn = Path(args.output_dir) / "model.pt"
        logger.info(f"saving model to {fn}") # Log the location of the saved model
        torch.save(model.state_dict(), str(fn)) # Save the model
        
    if args.save_code: # save_code = True -> True
        logger.info(f"saving code to {args.output_dir}") # Log the location of the saved code
        x = idx_w[X_val[0]]
        x = x[x != "<pad>"].tolist()
        try:
            code_utils.model_to_code(
                model=model,
                idx_w=idx_w,
                idx_t=idx_t,
                embed_csv=not args.one_hot_embed,
                unembed_csv=True,
                one_hot=args.one_hot_embed,
                autoregressive=args.autoregressive,
                var_types=True,
                output_dir=args.output_dir,
                name=args.dataset,
                example=x,
            )
        except Exception as e:
            logger.error(f"error saving code: {e}")

    return df


def run(args):
    set_seed(args.seed) # Set seed - Line X
    
    (
        train,
        test,
        val,
        idx_w,
        w_idx,
        idx_t,
        t_idx,
        X_train,
        Y_train,
        X_test,
        Y_test,
        X_val,
        Y_val,
    ) = data_utils.get_dataset(
        name=args.dataset,
        vocab_size=args.vocab_size,
        dataset_size=args.dataset_size,
        train_min_length=args.train_min_length,
        train_max_length=args.train_max_length,
        test_min_length=args.test_min_length,
        test_max_length=args.test_max_length,
        seed=args.seed,
        get_val=True,
        unique=args.unique,
    ) # Get dataset - data_utils, line 564
    
    # Log dataset information
    logger.info(f"vocab size: {len(idx_w)}")
    logger.info(f"X_train: {X_train.shape}, Y_train, {Y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, Y_val, {Y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, Y_test, {Y_test.shape}")
    a = set(["".join(s) for s in train["sent"]])
    b = set(["".join(s) for s in test["sent"]])
    logger.info(f"{len(a)}/{len(train)} unique training inputs")
    logger.info(f"{len(b - a)}/{len(test)} unique test inputs not in train")
    
    results = run_program(
        args,
        idx_w=idx_w,
        w_idx=w_idx,
        idx_t=idx_t,
        t_idx=t_idx,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        X_val=X_val,
        Y_val=Y_val,
    ) # Run program - Line X
    
    fn = Path(args.output_dir) / "results.csv"
    logger.info(f"writing results to {fn}")     # Log location of results
    results.to_csv(fn, index=False)             # Write results to csv file


if __name__ == '__main__':
    args = parse_args()                 # Parse arguments - Line X
    logger.info(f"args: {vars(args)}")  # Log arguments
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f)        # Write variables in arg.json file
    run(args)                           # Run the model - Line X