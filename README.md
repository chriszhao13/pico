# pico

---
Software requirement
-----
gcc
CUDA
Cmake

---
Hardware
------
Nvidia GPU:  GTX 3090 (tested)

----
Datasets
------
https://sparse.tamu.edu/
http://konect.cc/networks/

----
Compile
------
For index2core
```bash
cd  ./pico/index2core
mkdir&cd build
cmake ..
make
./HistoCore ./data/example.txt
```
For peel
   
```bash
#  For GPP vs PeelOne
cd  ./pico/peel
mkdir&cd build
cmake ..
make
./peelone ./data/example.txt
#  For Parallel Peel-dyn vs PeelOne-dyn
cd ./pico
make
./peelone_dyn  ./data/example.txt
```

----
Reference
-------

If you use Pico in your project, please cite the following paper.

@inproceedings{icde2024
under review
}
