from setuptools import setup, find_packages

setup(name= 'preprocessing',
      version= '0.1',
      description= """Preprocessing is a module created as an assignment for Data Mining Classes
      in Poznan University of Technology. It's goal is to provide neccessery tools to pre-proccess
      the data before using it as a training data for machine learning algorithms like dnn or som""",
      author= 'Data Dwarfes',
      packages=find_packages(),
      zip_safe= False)
