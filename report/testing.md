# Testing

Testing on 50 epochs with SGD

**params:** batch size = 64, learning rate = 0.01, decay rate = 0.05, L2 lambda = 0.01

```
Epoch 49 complete
Loss:1.3707806268319247
Validation Loss:1.4354981675006857
Classification accuracy: 0.5126
```

Testing on 50 epochs with Adam

**params:** batch size = 16, learning rate = 2e-5, decay rate = 0.001, L2 lambda = 0.001

```
Loss:1.5180683437788132
Validation Loss:1.5219432411465665
Classification accuracy: 0.4622
```

## Comparison

| SGD     [3072, 256, 10]                                                                                                  | Adam [3072, 256, 10]                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Loss:1.9325223558582274                                                                                                  | Loss:1.7086229728724305                                                                                                  |
| Validation Loss:1.794091993884861                                                                                        | Validation Loss:1.5163626392564575                                                                                       |
| Classification accuracy: 0.3798                                                                                          | Classification accuracy: 0.4682                                                                                          |
| Log: finished with params: epochs - 50, batch size - 16, learning rate - 2e-05, decay rate - 0.001, lambda for L2 - 0.01 | Log: finished with params: epochs - 50, batch size - 16, learning rate - 2e-05, decay rate - 0.001, lambda for L2 - 0.01 |


### Comparing

#### 1.
```
Classification accuracy: 0.4887
Log: finished with params: epochs - 20, batch size - 32, learning rate - 0.001, decay rate - 0.01, L2 enabling is False for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 10]
```

```
Classification accuracy: 0.4812
Log: finished with params: epochs - 20, batch size - 32, learning rate - 0.001, decay rate - 0.01, L2 enabling is True for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 10]
```

#### 2.

```
Classification accuracy: 0.5007
Log: finished with params: epochs - 30, batch size - 16, learning rate - 0.001, decay rate - 0.01, L2 enabling is True for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 100, 10]
```

```
Classification accuracy: 0.4983
Log: finished with params: epochs - 30, batch size - 16, learning rate - 0.001, decay rate - 0.01, L2 enabling is False for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 100, 10]
```

#### 3.

```
Classification accuracy: 0.3012
Log: finished with params: epochs - 30, batch size - 16, learning rate - 2e-05, decay rate - 0.01, L2 enabling is False for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 100, 10]
```
- ^ faster learning at start but then it stagnates

```
Classification accuracy: 0.303
Log: finished with params: epochs - 30, batch size - 16, learning rate - 2e-05, decay rate - 0.01, L2 enabling is True for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 100, 10]
```

#### 4.

```
Classification accuracy: 0.4479
Log: finished with params: epochs - 30, batch size - 16, learning rate - 2e-05, decay rate - 0.001, L2 enabling is True for 0.01
Log: network was optimized with adam optimizer and layers were [3072, 200, 100, 10]
```

```
Classification accuracy: 0.3468
Log: finished with params: epochs - 30, batch size - 16, learning rate - 2e-06, decay rate - 0.001, L2 enabling is True for 0.01
Log: network was optimized with adam optimizer and layers were [3072, 200, 100, 10]
```

This learning rate of 2e-05 is better


#### 5.

best results for adam with 2e-04 = 0.0002

```
Classification accuracy: 0.5087
Log: finished with params: epochs - 30, batch size - 16, learning rate - 0.0002, decay rate - 0.001, L2 enabling is True for 0.01
Log: network was optimized with adam optimizer and layers were [3072, 200, 100, 10]
```

With L2 enabled the loss was a little higher through the training but the validation loss was alright.
With it enabled we get the same result with 0.005 less accuracy, which is irrelevant

```
Classification accuracy: 0.5031
Log: finished with params: epochs - 30, batch size - 16, learning rate - 0.0002, decay rate - 0.001, L2 enabling is False for 0.01
Log: network was optimized with adam optimizer and layers were [3072, 200, 100, 10]
```


#### Random sgd

```
Classification accuracy: 0.4988
Log: finished with params: epochs - 30, batch size - 32, learning rate - 0.001, decay rate - 0.01, L2 enabling is True for 0.01
Log: network was optimized with sgd optimizer and layers were [3072, 200, 100, 10]
```


# 