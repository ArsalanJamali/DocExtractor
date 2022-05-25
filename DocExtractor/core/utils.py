from torch.nn import CrossEntropyLoss
from torch import device,load,tensor
from torch.cuda import is_available
from transformers import LayoutLMTokenizer
from transformers import LayoutLMForTokenClassification
import numpy as np
import pytesseract
from DocExtractor.settings import tesseract_location
import os


run_device = device("cuda" if is_available() else "cpu")
pytesseract.pytesseract.tesseract_cmd=tesseract_location
config_files_path='E:\\DocExtractor-backend\\DocExtractor\\DocExtractor\\config_files'

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

def model_load(PATH, num_labels,token_classification_file_name):
    model = LayoutLMForTokenClassification.from_pretrained(os.path.join(config_files_path,token_classification_file_name), num_labels=num_labels)
    model.load_state_dict(load(PATH, map_location=run_device))
    model.to(run_device)
    model.eval()
    return model

def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0]):
      width, height = image.size

      tokens = []
      token_boxes = []
      actual_bboxes = [] #Extra b because actual_boxes is already used
      token_actual_boxes = []
      for word, box, actual_bbox in zip(words, boxes, actual_boxes):
          word_tokens = tokenizer.tokenize(word)
          tokens.extend(word_tokens)
          token_boxes.extend([box] * len(word_tokens))
          actual_bboxes.extend([actual_bbox] * len(word_tokens))
          token_actual_boxes.extend([actual_bbox] * len(word_tokens))

      # Truncation: account for [CLS] and [SEP] with "- 2". 
      special_tokens_count = 2 
      if len(tokens) > args.max_seq_length - special_tokens_count:
          tokens = tokens[: (args.max_seq_length - special_tokens_count)]
          token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
          actual_bboxes = actual_bboxes[: (args.max_seq_length - special_tokens_count)]
          token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]

      # [SEP] token, with corresponding token boxes and actual boxes
      tokens += [tokenizer.sep_token]
      token_boxes += [sep_token_box]
      actual_bboxes += [[0, 0, width, height]]
      token_actual_boxes += [[0, 0, width, height]]
      
      segment_ids = [0] * len(tokens)

      # next:[CLS] token
      tokens = [tokenizer.cls_token] + tokens
      token_boxes = [cls_token_box] + token_boxes
      actual_bboxes = [[0, 0, width, height]] + actual_bboxes
      token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
      segment_ids = [1] + segment_ids

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = args.max_seq_length - len(input_ids)
      input_ids += [tokenizer.pad_token_id] * padding_length
      input_mask += [0] * padding_length
      segment_ids += [tokenizer.pad_token_id] * padding_length
      token_boxes += [pad_token_box] * padding_length
      token_actual_boxes += [pad_token_box] * padding_length

      assert len(input_ids) == args.max_seq_length
      assert len(input_mask) == args.max_seq_length
      assert len(segment_ids) == args.max_seq_length
      #assert len(label_ids) == args.max_seq_length
      assert len(token_boxes) == args.max_seq_length
      assert len(token_actual_boxes) == args.max_seq_length
      
      return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def iob_to_label(label):
  if label != 'O':
    return label[2:]
  else:
    return ""

def get_text(x,y,w,h,ocr_df):
  df=ocr_df[['left','top','width','height','text']]
  df = df.reset_index()  # make sure indexes pair with number of rows
  for index, row in df.iterrows():
    x2,y2,w2,h2,text=row['left'],row['top'],row['width'],row['height'],row['text']
    if(x==x2 and y==y2 and w==x2+w2 and h==y2+h2 ):
      return text
  return ''

def preprocess(output):
  output2=dict()
  for key in output.keys():
    value=output[key].split()
    value=(map(lambda x: x.lower(), value))
    value=" ".join(list(dict.fromkeys(value)))
    output2[key]=value
  return output2


def process_image(image,type):

    invoice_recipt_po_dict=''
    model_file_name=''
    label_file_name=''
    token_classification_filename=''

    if type==0:
        invoice_recipt_po_dict={
            'invoice':'',
            'due_date':'',
            'tax':'',
            'invoice_date':'',
            'total':'',
            'buyer_address':'',
            'seller_address':'',
            'seller_email':'',
            'seller_name':'',
            'seller_phone':'',
            'po_number':'',
            'buyer_name':'',
            'buyer_phone':'',
            'subtotal':'',
            'buyer_email':''
            }
        model_file_name='layoutlm_weights.pt'
        label_file_name='labels_invoice.txt'
        token_classification_filename='LayoutLMTokenClassification-Invoice'
    elif type==1:
        invoice_recipt_po_dict={
            'company':'',
            'date':'',
            'address':'',
            'total':''
        }
        model_file_name='layoutlm_weights_receipts.pt'
        label_file_name='labels_receipt.txt'
        token_classification_filename='LayoutLMTokenClassification-Receipt'
    elif type==2:
        invoice_recipt_po_dict={
            'company_address':'',
            'delivery_date':'',
            'issue_date ':'',
            'purchase_order_no':'',
            'tax':'',
            'total_amount':'',
            'vendor_address':'',
        }
        model_file_name='layoutlm_weights_PO_3.06.22.pt'
        label_file_name='labels_PO.txt'
        token_classification_filename='LayoutLMTokenClassification-PO'


    labels = get_labels(os.path.join(config_files_path,label_file_name))
    num_labels = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}

    pad_token_label_id = CrossEntropyLoss().ignore_index
    tokenizer = LayoutLMTokenizer.from_pretrained(os.path.join(config_files_path,"tokenizer"))

    args = {'local_rank': -1,
            'overwrite_cache': True,
            'data_dir': '/content/data/',
            'model_name_or_path':'microsoft/layoutlm-base-uncased',
            'max_seq_length': 512,
            'model_type': 'layoutlm',}

    # print("dhdkshdkshdksdnd")
    args = AttrDict(args)

    model_path=os.path.join(config_files_path,model_file_name)
    model=model_load(model_path,num_labels,token_classification_filename)


    width, height = image.size
    w_scale = 1000/width
    h_scale = 1000/height

    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')           
    ocr_df = ocr_df.dropna() \
                .assign(left_scaled = ocr_df.left*w_scale,
                        width_scaled = ocr_df.width*w_scale,
                        top_scaled = ocr_df.top*h_scale,
                        height_scaled = ocr_df.height*h_scale,
                        right_scaled = lambda x: x.left_scaled + x.width_scaled,
                        bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    

    words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
        actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 
        actual_boxes.append(actual_box)
    
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))

    
    input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_example_to_features(image=image, words=words, boxes=boxes, actual_boxes=actual_boxes, tokenizer=tokenizer, args=args)
    input_ids = tensor(input_ids, device=run_device).unsqueeze(0)
    attention_mask = tensor(input_mask, device=run_device).unsqueeze(0)
    token_type_ids = tensor(segment_ids, device=run_device).unsqueeze(0)
    bbox = tensor(token_boxes, device=run_device).unsqueeze(0)
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

    token_predictions = outputs.logits.argmax(-1).squeeze().tolist() # the predictions are at the token level
    # print(token_predictions)

    word_level_predictions = [] # let's turn them into word level predictions
    final_boxes = []
    for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
        if(tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, 
                                                            tokenizer.sep_token_id, 
                                                            tokenizer.pad_token_id]):
        # skip prediction + bounding box
            continue
        else:
            word_level_predictions.append(token_pred)
            final_boxes.append(box)


    output_dict=dict()
    for prediction, box in zip(word_level_predictions, final_boxes):
        predicted_label = iob_to_label(label_map[prediction]).lower()
        
        if predicted_label!='':
            text=get_text(box[0],box[1],box[2],box[3],ocr_df)
        
            if predicted_label not in output_dict:
                output_dict[predicted_label]=text
            else:
                if text!='':
                    output_dict[predicted_label]+=(' '+text)
    
    output_dict=preprocess(output_dict)

    for key in output_dict.keys():
        if type==1:
            invoice_recipt_po_dict[key]=output_dict[key]
        else:
            if 'value' in key: 
                invoice_recipt_po_dict[key[:-6]]=output_dict[key]
    
    return invoice_recipt_po_dict