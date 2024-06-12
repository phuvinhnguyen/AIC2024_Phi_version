from transformers import DataCollatorForLanguageModeling
import torch
from typing import List, Dict

class VidLangCollator(DataCollatorForLanguageModeling):
    inference: bool = False

    def list2dict(self, features: List[Dict]) -> Dict:
        inputs_embeds = []
        inputs_embeds_mask = []
        event_embeds = []
        events_embeds_mask = []
        input_ids = []
        attention_mask = []
        phase = []
        labels = []
        
        if not self.inference:
            for i in features:
                inputs_embeds.append(i['inputs_embeds'])
                inputs_embeds_mask.append(i['inputs_embeds_mask'])
                event_embeds.append(i['event_embeds'])
                events_embeds_mask.append(i['events_embeds_mask'])
                phase.append(i['phase'])
                input_ids.append(i['input_ids'])
                attention_mask.append(i['attention_mask'])
                labels.append(i['labels'])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'inputs_embeds': inputs_embeds,
                'inputs_embeds_mask': inputs_embeds_mask,
                'event_embeds': event_embeds,
                'events_embeds_mask': events_embeds_mask,
                'phase': phase,
                'labels': labels
            }
        else:
            for i in features:
                inputs_embeds.append(i['inputs_embeds'])
                inputs_embeds_mask.append(i['inputs_embeds_mask'])
                input_ids.append(i['input_ids'])
                attention_mask.append(i['attention_mask'])
                phase.append(i['phase'])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'inputs_embeds': inputs_embeds,
                'inputs_embeds_mask': inputs_embeds_mask,
                'phase': phase,
            }
    
    def create_pad(self, tensor: torch.Tensor, max_shape: torch.Size):
        tensor_shape = tensor.shape
        pad_raw = (torch.tensor(max_shape) - torch.tensor(tensor_shape)).tolist()
        result = []
        for i in pad_raw[::-1]:
            result += [0, i]
        return result

    def padd_tensor_for_batch(self, tensors, log=None):
        max_shape = torch.tensor([max(dim_size) for dim_size in zip(*[tensor.shape for tensor in tensors])])

        if not self.inference:
            list_of_tensors = [torch.nn.functional.pad(tensor, pad=self.create_pad(tensor, max_shape)).unsqueeze(0) for tensor in tensors]

            try:
                output = torch.cat(list_of_tensors, dim=0)
            except:
                print(log)
                print([i.shape for i in list_of_tensors])

            return output
        else:
            return torch.cat([torch.nn.functional.pad(tensor, pad=self.create_pad(tensor, max_shape)) for tensor in tensors], dim=0)
    
    def __call__(self, features: List[Dict], return_tensors=None) -> Dict:
        feature_batch = self.list2dict(features)
        
        return {key: self.padd_tensor_for_batch(value, key) for key, value in feature_batch.items()}