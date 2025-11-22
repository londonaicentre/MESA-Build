prettier:
	black --skip-string-normalization .

test:
	for env in `tox -l`; do echo $$env; tox -e $$env || break; done