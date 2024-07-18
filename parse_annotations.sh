cd datasets/breathingSet2/temp

txt=$(ls *.txt)
for file in $txt; do
  IFS=$'\t' # Set IFS to tab character
  newFile=${file//txt/csv}
  echo -n "" > "$newFile"
  while read -r -a line
  do
      finalChar=','
      if [[ "${line[2]}" == "exhale" ]]; then
        finalChar='\n'
        echo "nl"
      fi
      l1=${line[0]}
      l2=${line[1]}
      echo -n "$(awk "BEGIN {print $l1*1000}")," >> "$newFile"
      echo -n "$(awk "BEGIN {print $l2*1000}")$finalChar" >> "$newFile"
  done < $file
done

exit 0