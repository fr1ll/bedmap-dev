Ways you might run bedmap:
1. end-to-end
2. embed images first, then create viewer
3. update viewer with new UMAP parameters (pretty similar to #2)

embed images includes:
- validate images
- embed images

create viewer includes
- create thumbnails
- validate embeddings match images
    - embeddings may come from external source
    - or embeddings may come from previous bedmap
- create layouts
- create web assets

update viewer does same as above, but skips create thumbnails

create web assets includes:
- copy thumbnails
- create json data: layouts, name, icon, filepaths

