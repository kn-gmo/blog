# NeuralProcess Image Interpolation Demo

[blog](https://recruit.gmo.jp/engineer/jisedai/blog/neuralprocess-imgimp/)

[original paper](https://arxiv.org/abs/1807.01622)

## train
```
#MNIST
python np_fig -m train -dt mnist

#CELEBA
python np_fig -m train -dt celeba
```

## evaluate(make movie)
```
#MNIST
python np_fig -m evaluate -dt mnist -nc 100 
#CELEBA
python np_fig -m evaluate -dt celeba -nc 2000

```
![demo](https://github.com/kn-gmo/blog/blob/master/NeuralProcess/src/viz/mnist/sample_100.gif)





