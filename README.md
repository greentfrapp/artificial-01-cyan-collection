# CYAN COLLECTION

**(art)ificial** *project01*

<img src="https://raw.githubusercontent.com/greentfrapp/artificial-01-cyan-collection/master/CYAN%20COLLECTION/all.jpg" width="250" alt="CYAN COLLECTION"></img>

This repository contains the script that created [CYAN COLLECTION](https://github.com/greentfrapp/artificial-01-cyan-collection/tree/master/CYAN%20COLLECTION).

### Instructions

Run the following to generate an image with the default parameters. The default resulting image should be classified as 'strawberry' by Keras' ResNet50.

```
$ python3 main.py
```

**Parameters**

- ImageNet Class (by ID): `-class=949`
- Hue to apply (hex): `-hue=#00FFFD`
- Number of steps: `-steps=2560`
- Output image size (pixels): `-size=1024`
- Output file: `-path=image.jpg`

### Help

Run with the `--help` flag to get a list of possible flags.

```
$ python3 main.py --help
```
