touch breathingSet1/annotations.csv

cd breathingSet1

textFiles=$(ls *.txt)
for file in $textFiles; do
    if [[ $file = "annotations.txt" ]]; then
      break
    fi
    wavName=${file//.txt/.wav}
    echo -n "${wavName}" >> annotations.csv
    for element in $(cat $file); do
      if [[ $element = "0" ]]; then
        continue
      fi
      echo -n ",${element}" >> annotations.csv
    done
    echo "" >> annotations.csv
    # cat $file >> annotations.txt
done


# for downloading from gsutil: gsutil cp -r gs://verse-audio-collection.appspot.com "C:\Users\kiera\Documents\verse\breathNN\datasets"