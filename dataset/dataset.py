from torch.utils.data import Dataset
import glob
import os
import json
import math
import torch
import torch.nn.functional as F

class AICDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 captions_path,
                 videos_path,
                 num_embed=20,
                 is_test: bool=False,
                 dataset_type=0,
                 device='cpu'
                 ):
        '''
        dataset_type = 0: internal
        dataset_type = 1: external
        '''
        self.json_paths = glob.glob(f'{captions_path}/**/*.json', recursive=True)

        self.labels = [
            "pre-recognition",
            "recognition",
            "judgment",
            "action",
            "avoidance"
        ]
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.device = device
        self.num_embed = num_embed
        self.dataset_type = dataset_type
        self.dataset = [i for i in self.get_data(self.json_paths, videos_path)]
            
    def get_data(self, json_paths, video_path):
        mapping_tokens = {(i, j): [i * 2 + j] for i in range(5) for j in range(2)}
        mapping_events = {'prerecognition': 0, 'recognition': 1, 'judgement': 2, 'action': 3, 'avoidance': 4,
                          "0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
        mapping_objects = {'pedestrian': 0, 'vehicle': 1}
        
        for json_path in json_paths:
            # Read json caption file
            with open(json_path, 'r') as rf:
                captions = json.load(rf)
            
            # Get event phase
            event_phases = captions['event_phase']
            
            # Get videos dir
            if self.dataset_type == 0:
                videos = captions['overhead_videos'] if 'overhead_videos' in captions.keys() else [captions['vehicle_view']]
            else:
                videos = [captions['video_name']]
            videos = [torch.load(os.path.join(video_path, video.replace('mp4', 'pt')), map_location=torch.device(self.device)) for video in videos]

            max_num_frame = max([i.shape[0] for i in videos])
            input_video_mask_tensor = torch.tensor([i.shape[0] * [1] + [0] * (max_num_frame - i.shape[0]) for i in videos]).T

            videos = torch.concat([F.pad(i, (0,0,0, max_num_frame-i.shape[0])).unsqueeze(0) for i in videos], dim=0).permute(1,0,2)
            
            input_video_mask = input_video_mask_tensor.tolist()

            for event in event_phases:
                # Get label
                labels = mapping_events[event['labels'][0]]

                # Get start and end frame idx
                start_time, end_time = math.floor(float(event['start_time'])), math.ceil(float(event['end_time']))

                # Get and modify attribute
                if self.is_test:
                    caption_pedestrian = self.tokenizer(f"[pedestrian] [{self.labels[labels]}]: ")
                    caption_vehicle = self.tokenizer(f"[vehicle] [{self.labels[labels]}]: ")
                else:
                    caption_pedestrian = self.tokenizer(f"[pedestrian] [{self.labels[labels]}]: {event['caption_pedestrian']}<|eos|>")
                    caption_vehicle = self.tokenizer(f"[vehicle] [{self.labels[labels]}]: {event['caption_vehicle']}<|eos|>")
                
                # Get events
                event_embeds = videos[start_time:end_time+1,...]
                events_embeds_mask = input_video_mask_tensor[start_time:end_time+1,...].tolist()
                
                classify_token_pedestrian = mapping_tokens[(labels, mapping_objects['pedestrian'])]
                classify_token_vehicle = mapping_tokens[(labels, mapping_objects['vehicle'])]

                # create the result
                output_pedestrian = {
                    'input_ids': torch.tensor(caption_pedestrian['input_ids']),
                    'attention_mask': torch.tensor([1] * self.num_embed + caption_pedestrian['attention_mask']),
                    'labels': torch.tensor([0] * self.num_embed + caption_pedestrian['input_ids']),
                    'event_embeds': event_embeds,
                    'events_embeds_mask': torch.tensor(events_embeds_mask),
                    'inputs_embeds': videos,
                    'inputs_embeds_mask': torch.tensor(input_video_mask),
                    'phase': torch.tensor(classify_token_pedestrian)
                }
                
                output_vehicle = {
                    'input_ids': torch.tensor(caption_vehicle['input_ids']),
                    'attention_mask': torch.tensor([1] * self.num_embed + caption_vehicle['attention_mask']),
                    'labels': torch.tensor([0] * self.num_embed + caption_vehicle['input_ids']),
                    'event_embeds': event_embeds,
                    'events_embeds_mask': torch.tensor(events_embeds_mask),
                    'inputs_embeds': videos,
                    'inputs_embeds_mask': torch.tensor(input_video_mask),
                    'phase': torch.tensor(classify_token_vehicle)
                }

                # Continue if there are any problems
                if len(event_embeds.shape) != 3 or len(output_vehicle['events_embeds_mask'].shape) != 2:
                    continue

                if self.is_test:
                    # output pedestrian
                    output_pedestrian.update({'id': json_path})
                    output_pedestrian['inputs_embeds'] = torch.tensor(output_pedestrian['event_embeds']).unsqueeze(0).to(self.device)
                    output_pedestrian['inputs_embeds_mask'] = output_pedestrian['events_embeds_mask'].unsqueeze(0).to(self.device)
                    output_pedestrian['phase'] = output_pedestrian['phase'].unsqueeze(0).to(self.device)
                    output_pedestrian['input_ids'] = output_pedestrian['input_ids'].unsqueeze(0).to(self.device)
                    output_pedestrian['attention_mask'] = output_pedestrian['attention_mask'].unsqueeze(0).to(self.device)
                    output_pedestrian['time'] = torch.tensor([start_time, end_time]).unsqueeze(0).to(self.device)

                    output_pedestrian.pop('event_embeds')
                    output_pedestrian.pop('events_embeds_mask')
                    output_pedestrian.pop('labels')

                    # output vehicle
                    output_vehicle.update({'id': json_path})
                    output_vehicle['inputs_embeds'] = torch.tensor(output_vehicle['event_embeds']).unsqueeze(0).to(self.device)
                    output_vehicle['inputs_embeds_mask'] = output_vehicle['events_embeds_mask'].unsqueeze(0).to(self.device)
                    output_vehicle['phase'] = output_vehicle['phase'].unsqueeze(0).to(self.device)
                    output_vehicle['input_ids'] = output_vehicle['input_ids'].unsqueeze(0).to(self.device)
                    output_vehicle['attention_mask'] = output_vehicle['attention_mask'].unsqueeze(0).to(self.device)
                    output_vehicle['time'] = torch.tensor([start_time, end_time]).unsqueeze(0).to(self.device)
                    
                    output_vehicle.pop('event_embeds')
                    output_vehicle.pop('events_embeds_mask')
                    output_vehicle.pop('labels')

                yield output_pedestrian
                yield output_vehicle


                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    


class CombineAICDataset(Dataset):
    def __init__(self, datasets = []):
        self.datasets = datasets
        self.indexs = [len(i) for i in datasets]
        self.len = sum([len(i) for i in datasets])
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data, index = self.datasets[0], 0

        for i in self.datasets:
            if index + len(i) > idx:
                data = i
                index = idx - index
                break
            else:
                index += len(i)

        return data[index]