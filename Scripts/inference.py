from utils import *
from model import get_model

base_path = "./test/image/"
pred = []
submission = pd.DataFrame()
ID = []
seg_masks = []

transform = A.Compose([
    ToTensorV2()
])
model = get_model()
model.load_state_dict(torch.load('./best_unet_model.pth'))
model.eval()
for step, files in enumerate(tqdm(os.listdir(base_path))):
    img_path = os.path.join(base_path, files)
    name = files.split('.')[0]
    array = tile_mask(files, base_path, 128)
    output_masks = []
    for i, j, patches in array:
        img_array = np.array(patches).astype('float32')
        
        img_array = transform(image = img_array)['image'].unsqueeze(0)
        op = model(img_array)
        mask_1 = reverse_one_hot((torch.sigmoid(op[0])).float().permute(1,2,0).cpu().detach().numpy())

        output_masks.append((i, j, mask_1))
    img = join_tiles_np(output_masks, w, h, d = 128)
    
    labeled_mask = label(img)
    polygons, _ = mask_to_polygons(mask = labeled_mask)

    def plot_polygons(polygons, title):
        coordinates = []
        for polygon in polygons:
            if polygon.is_empty:
                continue
            exterior_coords = np.array(polygon.exterior.coords)
            coordinates.append((exterior_coords))
        return coordinates
    
    coordinates = plot_polygons(polygons, title = 'Image')
    outputs = [list(map(tuple, coordinate)) for coordinate in coordinates]
    ID.append(name)
    seg_masks.append(str(outputs))


submission['ImageID'] = ID
submission['Coordinates'] = seg_masks
submission.to_csv('submission.csv', index = False)