init:
	uv sync
	. ./.venv/bin/activate && ansible-galaxy install -r requirements.yml
	docker context use default
	docker compose build

up:	
	docker context use default
	docker compose push
	terraform init
	terraform plan
	terraform apply
	. ./.venv/bin/activate && ansible-playbook site.yml
	ssh-keyscan -t ed25519 $$(terraform output -raw dns) >> ~/.ssh/known_hosts
	docker context create genollama --docker "host=ssh://ubuntu@$$(terraform output -raw dns)"
	docker context use genollama
	docker compose up -d

down:
	docker context rm -f genollama
	terraform destroy

prettier:
	terraform fmt -recursive
	npx prettier --write "**/*.{yml,yaml}"