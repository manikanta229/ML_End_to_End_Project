#setup.py is used to built your ML application as a 
from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    e_minu = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if e_minu in requirements:
            requirements.remove(e_minu)


setup(
    name = 'END_To_END_ML_Project',
    version='0.0.1',
    author = "mani",
    author_email='manikantaalluri229@gmail.com',
    packages = find_packages(),
    install_require = get_requirements('requirements.txt')
)