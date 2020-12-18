# This MakeFile is compatible with py-make: https://github.com/tqdm/py-make

VIRTUALENV_NAME = d2v_env
PROJECT_NAME = dater_to_vec

## setup - install pre-commit hooks.
install:
	pre-commit install


## lint - Lint the project
lint:
	pre-commit run --all-files

## test - Test the project
test:
	pytest ./dater_to_vec

## coverage - Test the project and generate an HTML coverage report
coverage:
	pytest --cov=$(PROJECT_NAME) --cov-branch --cov-report=html

.PHONY: install lint test coverage


### Cheat Sheet

## venv - Install the virtual environment - WARNING: you would need py-make in base environment
# venv:
# 	conda env create -f $(VIRTUALENV_NAME).yml

# ## activate_venv:- Activate the virtual environment -  WARNING: you would need py-make in base environment
# activate_venv:
# 	conda activate $(VIRTUALENV_NAME)

# ## deactivate_venv: Deactivate the virtual environment -  WARNING: you would need py-make in base environment
# deactivate_venv:
# 	conda deactivate

## delete - Remove the virtual environment and clear the data reposiroty
## add /Q to be not require autho
# delete:
# 	conda env remove --name $(VIRTUALENV_NAME)
# 	DEL /S "./data/*"
