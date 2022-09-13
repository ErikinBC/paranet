# # in main folder
# rm dist/* 
# python setup.py bdist_wheel --universal

# # on some test conda env
# twine upload --repository-url https://test.pypi.org/legacy/ dist/paranet*
# pip uninstall paranet
# pip install --index-url https://test.pypi.org/simple/ paranet --user

# # Upload to PyPI: https://pypi.org/project/paranet/
# twine upload dist/paranet*
# pip uninstall paranet
# pip install paranet
