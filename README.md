# pytorch

* Pull the docker image
    ```sh
    $ docker pull pytorch/pytorch:latest
    ```

* In the project root dir
    ```
    $ docker build -t pytorch .    
    ```

* set up environment
    ```sh
    $ source setup.sh
    ```

* Run
    ```sh
    dkb python3 classifier.py
    ```