# Conda
create
```shell
conda create --name=yolo7 python=3.6.9 -y 
```
remove conda env
```shell
conda env remove --name yolo7 -y
```

add jupyter kernel
```shell
python -m ipykernel install --user --name yolo7 --display-name "YOLO V7 MOT"
```
remove jupyter kernel
```shell
jupyter kernelspec uninstall yolo7 -y
```