# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# This file specifies the steps to run and their order and allows running them.
# Type `make` for instructions. Type make <command> to execute a command.
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.DEFAULT_GOAL := help

NAME=$(shell basename `pwd`)
SAMPLES=$(shell ls data)

help:  ## Display help and quit
	@echo Makefile for the $(NAME) project/package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

requirements:  ## Install Python requirements
	pip install -r requirements.txt

transfer:  ## [dev] Transfer data from wcm.box.com to local environment (to run internally at WCM)
	imctransfer -q 2021  # Query for files produced in 2021 only

backup_time:
	echo "Last backup: " `date` >> _backup_time
	chmod 700 _backup_time

_sync:
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/$(NAME)

sync: _sync backup_time ## [dev] Sync data/code to SCU server


upload_data: ## [dev] Upload processed files to Zenodo
	@echo "Warning: this step is not meant to be run, but simply details how datasets were uploaded."
	python -u src/_upload.py

download_data: ## [TODO!] Download processed data from Zenodo (for reproducibility)
	@echo "Not yet implemented!"
	python -u src/_download_datasets.py

analysis:  ## Run the actual analysis
	@echo "Running analysis!"
	python -u src/analysis.py

figures:  ## Produce figures in various formats
	cd figures; ./process.sh


.PHONY : help \
	requirements \
	transfer \
	sync \
	upload_data \
	download_data \
	analysis \
	figures
