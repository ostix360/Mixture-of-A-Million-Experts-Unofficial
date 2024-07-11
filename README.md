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

## Tests

### Test 1

The model has 22.2M parameters

The first test doesn't use glu and uses num_experts_per_tok=30, num_local_experts=36.

Train loss at the last step is 4.58 and train loss given by transformer's trainer 5.42

### Test 2

The model has 22.3M parameters

The second test uses glu and uses num_experts_per_tok=26, num_local_experts=30.

Train loss at the last step is 4.53 and train loss given by transformer's trainer 5.34


### Test 3 (baseline)

The model has 22.0M parameters

The third test uses glu and uses num_experts_per_tok=2, num_local_experts=4. (intermediate_size=256)

Train loss at the last step is 4.49 and train loss given by transformer's trainer 5.35

### Conclusion

The tow first test consume a lot more memory i.e. 3 times at least than the third test. The third test is the most efficient in terms of memory consumption and computation.
And also in term of time (7 times)

## TODO

- [x] Implement and test PEER layer
- [x] Add batch norm
- [x] Test training
- [x] Make it as package
- [ ] Add custom kernel
- [x] Add GLU / gate to the PEER layer
- [x] Validate the implementation by testing base line


## Contributing

Contributions are welcome.

Please open a pull request with the proposed changes.

## References

- [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153)
- [PKM (for key and indices handling)](https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb)

