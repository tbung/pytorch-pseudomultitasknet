# pseudo-multitask learning

Attempts to create an interesting feature space using multiple tasks for one
single network. Includes a reversible and thereby generative model.

## Requirements

- pytorch
- tqdm
- numpy

## Running in a pytorch docker container

Just in case I ever forget or someone else wants to run this without worries.

```
docker run --runtime=nvidia -it -v $(pwd):/workspace pytorch_cuda9:latest zsh
```

## TODO

- [ ] Add generative side loss
- [x] Take a look at interpolation between samples
- [ ] Further improve net architecture
- [ ] Noise regularization
- [ ] SVD sparsity loss
- [ ] Learning by assoziation
