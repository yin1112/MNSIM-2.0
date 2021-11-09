#!/bin/bash
# Ellis Brown

start=`date +%s`

cd ./data/

mkdir -p voc2007
cd voc2007

if [ ! -d "/VOCdevkit" ]; then
  exit
fi

echo "Downloading VOC2007 trainval ..."
# Download the data.
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
echo "Downloading VOC2007 test data ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar
echo "removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"