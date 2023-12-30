export FILENAME=libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
sudo tar -C /usr/local -xzf ${FILENAME}

if use "/usr/local/lib"
                
                sudo ldconfig /usr/local/lib

else
                
                export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib


target_link_libraries(testtryfinally PRIVATE sharedlibs tensorflow)