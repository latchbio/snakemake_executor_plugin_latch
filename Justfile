@build:
  rm -rf dist
  poetry build

@publish:
  poetry publish -u __token__ -p "$(<credentials/pypi_token)"
  rm -rf dist
