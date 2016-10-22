cd ../..
python -c 'import crowddata.questions.bird_search_data as BSD; BSD.make_birds_flickr_dataset()'
cd $SPARROW_FLICKR_WEB
chmod a+x images
chmod a+r images/*.jpg
