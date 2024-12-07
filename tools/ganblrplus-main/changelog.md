# Notes

1. Depends on pyitlit package, which on its last version has problems with numpy. To install a previous pyitlit version, needed to download the package and install locally because of a comma missing in the setup file. Also did not work

2. First install ganblr and then install from local package pyitlib==0.2.2, where I have already corrected the missing comma

3. The code assumes that the target is always the last column - important!

4. GANBLR++ Creates all columns, even if not included into the "numerical_columns", as a number with decimals. 