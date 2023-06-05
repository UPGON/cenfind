version:
	@poetry version $(v)
	@git add pyproject.toml
	@git commit -m "v$(poetry version -s)"
	@git tag v$(poetry version -s) -m "v$(poetry version -s)"
	@git push
	@git push origin --follow-tags
	@poetry version

test:
	pytest tests -W ignore::DeprecationWarning
