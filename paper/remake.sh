#!/usr/bin/env bash

file=$1
cmd=$2
hash=`md5 ${file}`

while true :
do
  if [ "${hash}" != "`md5 ${file}`" ]
  then
    hash=`md5 ${file}`
    echo "<============ CHANGE DETECTED"
    ${cmd} 1> /dev/null
    echo "============> SCRIPT RE-EXECUTED"
  else
    sleep 1
  fi
done 

