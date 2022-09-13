from wbia.detecttools.directory import Directory
import utool as ut
import numpy as np
from os.path import abspath, join, split
import json
import wbia  # NOQA
import tqdm
import cv2


ibs = None  # NOQA


'''bash (host machine)

cd /datadrive

mkdir -p $(pwd)/wbia

cd $(pwd)/wbia

echo """
HOST_UID=$(id -u)
HOST_USER=$(whoami)
""" > $(pwd)/wbia.env

docker run \
    -d \
    -p 5000:5000 \
    --name wbia.ggr \
    -v $(pwd)/db:/data/db \
    -v $(pwd)/cache:/cache \
    -v /lynx-pie/wbia_pie_v2/reid-data/lynxcoco:/data/lynxcoco \
    -v /lynx-pie/wbia_pie_v2/lynxcoco/annotations:/data/annotations \
    --env-file $(pwd)/wbia.env \
    --restart unless-stopped \
    wildme/wbia:latest

sudo chown -R azureuser:azureuser .

docker exec -it wbia.ggr embed
'''


def background():
    """
    # NOQA
    cd ~/Downloads
    rsync -aP \
        anton:/lynx-pie/wbia_pie_v2/reid-data/lynxcoco/images/ \
        $(pwd)/wbia/import/
    rsync -aP \
        anton:/lynx-pie/wbia_pie_v2/lynxcoco/annotations/instances_train_right.json \
        $(pwd)/wbia/import/
    rsync -aP \
        anton:/lynx-pie/wbia_pie_v2/lynxcoco/annotations/instances_test_right.json \
        $(pwd)/wbia/import/
    rsync -aP \
        $(pwd)/wbia/db/output/ \
        anton:/lynx-pie/wbia_pie_v2/reid-data/lynxcoco/images/train_right_background/
    """ 
    IMPORT_PATH = '/data/import/'

    direct = Directory(IMPORT_PATH, recursive=True, images=True)
    filepaths = list(direct.files())

    gids = ibs.get_valid_gids()
    processed = set(ibs.get_image_uris_original(gids))

    filepaths_ = sorted(list(set(filepaths) - processed))
    chunks = ut.ichunks(filepaths_, 16)

    for filepath_chunk in tqdm.tqdm(chunks):
        try:
            gid = ibs.add_images(filepath_chunk)
            print(filepath_chunk)
            print(gid, ibs.get_image_uris_original(gid))
        except Exception:
            pass

    gids = ibs.get_valid_gids()
    uris_original = ibs.get_image_uris_original(gids)
    filenames_original = [split(uri)[1] for uri in uris_original]
    filenames_dict = dict(zip(filenames_original, gids))
    filenames_dict_ = dict(zip(gids, filenames_original))

    # Import existing JSON files
    train_coco_json = '/data/import/instances_train_right.json'
    test_coco_json = '/data/import/instances_test_right.json'

    for coco_filepath in [train_coco_json, test_coco_json]:
        with open(coco_filepath) as coco_file:
            coco_dict = json.load(coco_file)

        image_list = coco_dict.get('images', [])
        id_dict = {}
        image_dict = {}
        for image in image_list:
            # Get file name
            image_filename = image['file_name']
            image_filepath = abspath(join(
                IMPORT_PATH, 'train_right', image_filename
            ))
            image_id = image['id']

            image_dict[image_filename] = {
                'id': image_id,
                'filepath': image_filepath,
            }
            id_dict[image_id] = image_filename

        print(len(image_list))

        add_gids = []
        add_bbox = []
        for annotation in coco_dict.get('annotations', []):
            # annot_id = annotation['id']
            annot_bbox = annotation['bbox']
            image_id = annotation['image_id']
            try:
                assert image_id in id_dict
                image_filename = id_dict[image_id]
                assert image_filename in image_dict
                assert image_filename in filenames_dict
            except AssertionError:
                continue
            gid = filenames_dict[image_filename]
            xtl, ytl, w, h = annot_bbox
            xtl = int(np.around(xtl))
            ytl = int(np.around(ytl))
            w = int(np.around(w))
            h = int(np.around(h))
            add_gids.append(gid)
            add_bbox.append((xtl, ytl, w, h))

        print(len(add_gids))

        ibs.add_annots(add_gids, add_bbox)

    aids = ibs.get_valid_aids()
    ibs.set_annot_species(aids, ['lynx'] * len(aids))

    masks = ibs.get_annot_probchip_fpath(aids)  # NOQA
    images = ibs.get_images(gids)  # NOQA

    THRESH = 256 * 0.5

    for gid in tqdm.tqdm(gids):
        image = ibs.get_images(gid)
        aids = ibs.get_image_aids(gid)

        overlay = np.zeros(image.shape, dtype=np.float32)
        for aid in aids:
            mask = cv2.imread(ibs.get_annot_probchip_fpath(aid))
            xtl, ytl, w, h = ibs.get_annot_bboxes(aid)
            mask = cv2.resize(mask, (w, h))
            overlay[ytl:ytl+h, xtl:xtl+w, :] = mask

        cv2.imwrite('/data/db/raw/raw.%d.original.jpg' % (gid, ), overlay)

        overlay[overlay < THRESH] = 0
        overlay = cv2.dilate(overlay, (15, 15), iterations=5)
        overlay = cv2.blur(overlay, (30, 30))

        canvas = image.astype(np.float32)
        canvas = canvas * (overlay / 255.0)
        canvas = np.around(canvas)
        canvas[canvas < 0] = 0
        canvas[canvas > 255] = 255
        canvas = canvas.astype(np.uint8)

        cv2.imwrite('/data/db/raw/raw.%d.jpg' % (gid, ), image)
        cv2.imwrite('/data/db/raw/raw.%d.overlay.jpg' % (gid, ), overlay)
        cv2.imwrite('/data/db/raw/raw.%d.mask.jpg' % (gid, ), canvas)

        cv2.imwrite('/data/db/output/%s' % (filenames_dict_[gid]), canvas)
        cv2.imwrite('/data/db/output/%s.original.jpg' % (
            filenames_dict_[gid].replace('.', '-')
            ), 
            image
        )


def hotspotter():
    IMPORT_PATH = '/data/lynxcoco/images/train_new/'

    direct = Directory(IMPORT_PATH, recursive=True, images=True)
    filepaths = list(direct.files())

    gids = ibs.get_valid_gids()
    processed = set(ibs.get_image_uris_original(gids))

    filepaths_ = sorted(list(set(filepaths) - processed))
    chunks = ut.ichunks(filepaths_, 1)

    for filepath_chunk in tqdm.tqdm(chunks):
        try:
            gid = ibs.add_images(filepath_chunk)
            print(filepath_chunk)
            print(gid, ibs.get_image_uris_original(gid))
        except Exception:
            pass

    gids = ibs.get_valid_gids()
    uris_original = ibs.get_image_uris_original(gids)
    filenames_original = [split(uri)[1] for uri in uris_original]
    filenames_dict = dict(zip(filenames_original, gids))
    filenames_dict_ = dict(zip(gids, filenames_original))

    # Import existing JSON files
    train_coco_json = '/data/annotations/instances_train_new.json'
    test_coco_json = '/data/annotations/instances_val_new.json'

    train_gids = []
    test_gids = []
    for coco_filepath in [train_coco_json, test_coco_json]:
        with open(coco_filepath) as coco_file:
            coco_dict = json.load(coco_file)

        image_list = coco_dict.get('images', [])
        id_dict = {}
        image_dict = {}
        for image in image_list:
            # Get file name
            image_filename = image['file_name']
            image_filepath = abspath(join(
                IMPORT_PATH, 'train_right', image_filename
            ))
            image_id = image['id']

            image_dict[image_filename] = {
                'id': image_id,
                'filepath': image_filepath,
            }
            id_dict[image_id] = image_filename

        print(len(image_list))

        add_gids = []
        add_bbox = []
        add_names = []
        add_views = []
        for annotation in coco_dict.get('annotations', []):
            # annot_id = annotation['id']
            annot_bbox = annotation['bbox']
            name = annotation['attributes']['individual_id'].lower()
            viewpoint = annotation['viewpoint'].lower()
            assert viewpoint in ['left', 'right']
            image_id = annotation['image_id']
            try:
                assert image_id in id_dict
                image_filename = id_dict[image_id]
                assert image_filename in image_dict
                assert image_filename in filenames_dict
            except AssertionError:
                continue
            gid = filenames_dict[image_filename]
            xtl, ytl, w, h = annot_bbox
            xtl = int(np.around(xtl))
            ytl = int(np.around(ytl))
            w = int(np.around(w))
            h = int(np.around(h))
            if '_train_' in coco_filepath:
                train_gids.append(gid)
            if '_val_' in coco_filepath:
                test_gids.append(gid)
            add_gids.append(gid)
            add_bbox.append((xtl, ytl, w, h))
            add_names.append(name)
            add_views.append(viewpoint)

        print(len(add_gids))

        add_aids = ibs.add_annots(add_gids, add_bbox)
        ibs.set_annot_names(add_aids, add_names)
        ibs.set_annot_viewpoints(add_aids, add_views)

    aids = ibs.get_valid_aids()
    ibs.set_annot_species(aids, ['lynx'] * len(aids))

    # Start Hotspotter
    # Run code in feasability.py


if __name__ == '__main__':
    background()
    hotspotter()
