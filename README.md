# Mixture-of-A-Million-Experts-Unofficial
Trying to implement https://arxiv.org/abs/2407.04153

## Quick Start

### Install the package

```bash
pip install git+https://github.com/ostix360/Mixture-of-A-Million-Experts-Unofficial.git
```

### Test the implementation

Clone the repository:

```bash
git clone https://github.com/ostix360/Mixture-of-A-Million-Experts-Unofficial.git
```

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

Run the following command to test the implementation:

```bash
python test.py
```

## TODO

- [x] Implement and test PEER layer
- [x] Add batch norm
- [ ] Test training
- [x] Make it as package
- [ ] Add custom kernel
- [ ] Add GLU / gate to the PEER layer

## Contributing

Contributions are welcome.

Please open a pull request with the proposed changes.

## References

- [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153)
- [PKM (for key and indices handling)](https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb)

