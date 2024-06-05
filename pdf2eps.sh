for f in *.pdf
do 
    gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile=${f::-4}.eps $f
done
