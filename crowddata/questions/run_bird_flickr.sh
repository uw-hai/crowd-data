cd ../..
python -m crowddata.questions.bird_flickr_data
cd $SPARROW_FLICKR_WEB
chmod a+x images
chmod a+r images/*.jpg
