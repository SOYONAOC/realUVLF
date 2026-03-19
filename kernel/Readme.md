### Prepare
modify the ssh config

add the  `LocalForward $PORT localhost:$PORT`

such as my port is set to 1695, so my config is 
```
Host sc
  HostName 192.168.xx.xx
  User xxxxxx
  LocalForward 1695 localhost:1695
```

### First Step
run the command:
```bash
chmod +x startALl.sh
```
### Second Step
modify the file `startALl.sh`
#### 1. set the variable `PORT` to the port number you want to use.(e.g. `1695`, a best set is a non trivial digit number because it have low chance to be used by other programs e.g.8080 is used by other programs. An important item should be noticed is the `PORT` in this step should be same with the `Prepare` Step)
#### 2. modify the python envirnment path in line 13 of amd.sh

#### 3. inspect whether your envirnment have enough package used in this method, such as `jupyter`, `jupyterlab`, `notebook`, `ipykernel`, ``

### Third Step
run the command:
```bash
./startALl.sh
```
### Last Step
copy the file amd_jupyter.txt and find the line:
```bash
Then open in browser: xxxxxx
```
copy full of the http link (contains the token) and paste it in your browser.(it should be line 5)
Then you can use the jupyter notebook in your browser!
