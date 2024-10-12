# Shell and Automation Scripts
> Useful automation scripts.

## Installation

Installation
OS X & Linux:

Clone or download the repository.
HTTPS:
```sh
cd <parent directory>
git clone https://github.com/alexKotz-koz/Automation.git
```
---
Set up miniconda environment (*required for create_python.sh*):
- If miniconda is not installed on the local machine, please follow the steps outlined here before continuing: [Miniconda installation](https://docs.anaconda.com/free/miniconda/)
- If the miniconda base environment is not active, run this command:
    ```sh
    conda activate conda_env
    ```
    - **See .zshrc if you would like to create an alias to activate and deactivate a conda environment**. This can be useful if you do not want to have an active conda environment in your bash profile (i.e. everytime you open up a new shell instance).



## Usage example

Make each file an executable:

Linux/Unix
```sh
chmod +x <filename>
```
---
**create_python.sh:**

Usage: ./create_python.sh <project_name> <destination_directory> <project_type: 'web', 'standard', 'ml'>

$1 project_name | Desired name of the parent directory for the project <br>
$2 destination_directory | File path to the desired location of the project directory <br>
$3 project_type {'web', 'standard', 'ml'}<br>
- web: frontend and backend with a jupyter notebook under the frontend directory, and a flask installation
- standard: standard boilerplate for python project
- ml: (Machine Learning) standard boilerplate for python project with xlsxwriter, shelve, and matplotlib
```sh
./create_python.sh test_project ~/Desktop/ standard

```


## Contributing

1. Fork it (<https://github.com/alexKotz-koz/Automation.git>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

