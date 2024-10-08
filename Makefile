VENV := multi_env

.PHONY: setup install_lib clean fclean

all: setup install_lib

setup:
	python3 -m venv $(VENV)
	@echo "Virtual environment $(VENV) created."

install_lib:
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Installed lib."

clean:
	rm -rf *.npz
	@echo "Clean ok."

fclean: clean
	rm -rf $(VENV) dist
	@echo "Fclean ok."

re: fclean all
	@echo "Re ok."