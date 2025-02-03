import parselmouth
import numpy as np
import os
import logging
import json
import csv
import random
import shutil

from speechbrain.inference.speaker import EncoderClassifier
from hyperpyyaml import load_hyperpyyaml
from omegaconf import OmegaConf
import torch

#from spkanon_eval.featex.spkid.train import SpeakerBrain, prepare_dataset
from spkanon_eval.component_definitions import InferComponent

LOGGER = logging.getLogger("progress")


class ProsodyEmbedding(InferComponent):

    def __init__(self, config: OmegaConf, device: str) -> None:
        """
        diverse Parameter:
        sample_rate
        """
        self.device = device
        self.config = config
        self.pitch_floor = config.pitch_floor
        self.pitch_ceiling = config.pitch_ceiling
        self.window_length = config.window_length
        self.time_step = config.time_step
        self.minimum_intensity = config.minimum_intensity
        #self.feature_segments = config.get("feature_segments", {'f0':'first', 'energy':''})
        self.sampling_frequency = config.sampling_frequency
        #default_segments = {'f0': 'all', 'energy': 'all'}
        self.feature_segments = config.get("feature_segments")

    #Sagt aus, ob das Modell auf der CPU oder GPU läuft
    def to(self, device: str) -> None:
        self.device = device
        #self.model.to(device)

    
    @torch.inference_mode()
    def run(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Return speaker embeddings for the given batch of utterances.

        Args:
            batch: A list of three tensors in the following order:
            1. waveforms with shape (batch_size, n_samples)
            2. waveform lengths with shape (batch_size), as integers
            3. waveform speaker IDs with shape (batch_size), as integers

        Returns:
            A tensor containing the prosody embeddings with shape
            (batch_size, embedding_dim).
        """

        # Tensor printen
        # for i, tensor in enumerate(batch):
        #     print(f"Tensor {i} shape:", tensor.shape)
        #     print(f"Tensor {i} Inhalt:", tensor)
        #     print(f"Tensor {i} Datentyp:", tensor.dtype)

        waveforms = batch[0].cpu().numpy()
        
        embeddings = []

        #print("\nNach Konvertierung zu NumPy:")
        #print("Waveforms shape:", waveforms.shape)
        #print("Erste Waveform (ersten 10 Werte):", waveforms[0][:10])
        for waveform in waveforms:

            sound = parselmouth.Sound(values=waveform, sampling_frequency=self.sampling_frequency)
            
            embedding = self.create_embedding(sound)
            embeddings.append(embedding)

            
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(batch[0].device)
        print(f"output shape: {embeddings_tensor.shape}")
        print(f"output: {embeddings_tensor}")
        return embeddings_tensor



    #Vllt das globale Segment als ersts einfügen
    def segment_audio(self, audio):
        #Global Relative Time Intervals Approach
        # Gesamtlänge berechnen
        audio_data = audio.values[0]
        total_length = len(audio_data)
        segments = []

        # In 3 gleiche Segmente aufteilen
        segment_length = total_length // 3
        
        for i in range(3):
            start = i * segment_length
            end = start + segment_length
            segment_data = audio_data[start:end]
            segment_sound = parselmouth.Sound(segment_data, sampling_frequency=self.sampling_frequency)
            segments.append(segment_sound)
            #segments.append(segment)

        #Gesamte utterance als letztes Segment hinzufügen
        segments.append(audio)

        return segments


    
    def extract_f0(self, segment):
        #F0 extraction based on default values or dynamic pitch floor and ceiling
        #for testing
        method = "default_values"
        if method == "dynamic_values":
            pitch_floor, pitch_ceiling = self.get_boundaries( segment)
        else: 
            pitch_floor = self.pitch_floor
            pitch_ceiling = self.pitch_ceiling

        LOGGER.debug(f"Using pitch boundaries: {pitch_floor}-{pitch_ceiling}")

        sound = parselmouth.Sound(segment, sampling_frequency=self.sampling_frequency)
        pitch = sound.to_pitch_ac(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling, time_step=self.time_step, max_number_of_candidates=4, silence_threshold=0.05, 
                                  octave_cost=0.01, very_accurate=True, voicing_threshold=0.45)
        pitch_values = pitch.selected_array['frequency']

        return pitch_values

    #based on Speech Prosody: From Acoustics to Interpretation
    def get_boundaries(self, segment):
        sound = parselmouth.Sound(segment)
        #values need to be adjusted
        pitch = sound.to_pitch(pitch_floor=50, pitch_ceiling=300)
        values = pitch.selected_array['frequency']
        valid_values = values[values > 0]

        #print("Valid F0 values range:", np.min(valid_values), "-", np.max(valid_values))
    

        q1 = np.percentile(valid_values, 25) 
        print("Q1:", q1)

        pitch_floor = 0.75 * q1
        max_interval = 1.5
        pitch_ceiling = pitch_floor * (2 ** max_interval)
        print(pitch_floor, pitch_ceiling)

        return pitch_floor, pitch_ceiling


    def extract_energy(self, segment):
        sound = parselmouth.Sound(segment)
        intensity = sound.to_intensity(minimum_pitch=self.pitch_floor, subtract_mean=True, time_step = self.time_step) #minimum_pitch=100.0, time_step=self.time_step, 
        intensity_values = intensity.values[0]
        #rms = sound.get_root_mean_square()
        #linear_energy = 10 ** (intensity_values / 10)
        #return intensity_values
        return intensity_values


    def prosody_extraction(self, segment):
        f0 = self.extract_f0(segment)
        energy = self.extract_energy(segment)


        features = {
            'f0' : f0,
            'energy' : energy,
           # 'duration' : duration
        }
        return features


    class Aggregation:
        def __init__(self):
            self.aggregation_function = {
                'f0' : self.aggregate_f0,
                'energy' : self.aggregate_energy
            }


        @staticmethod
        def log_function(value):
            return np.log(value)

        @staticmethod
        def delta_log(value):
            return np.diff(ProsodyEmbedding.Aggregation.log_function(value))
        
        @staticmethod
        def amplitude_tilt(f0_contour):
            # Nullwerte entfernen
            f0_contour = f0_contour[f0_contour != 0]

            if len(f0_contour) == 0:
                return 0
            

            # Finde F0 Peak Index
            f0_peak_location = np.argmax(f0_contour)

            if f0_peak_location > 0:
                rise_segment = f0_contour[:f0_peak_location]
                rise = f0_contour[f0_peak_location] - np.min(rise_segment) if len(rise_segment) > 0 else 0
            else:
                rise = 0
            
            fall = f0_contour[f0_peak_location] - np.min(f0_contour[f0_peak_location:])

            if (abs(rise) + abs(fall)) == 0:
                return 0

            #Amplituden Tilt berechnen
            tilt= (abs(rise) - abs(fall)) / (abs(rise) + abs(fall))
            # Auf 4 Nachkommastellen runden
            return round(tilt, 4)  


        def aggregate_f0(self, values):
            values = np.array(values)
            valid_values = values[values> 0]
            if len (valid_values) == 0:
                return [0, 0, 0, 0]
            tilt = ProsodyEmbedding.Aggregation.amplitude_tilt(values)
    
            try:
                return [
                    round(np.mean(valid_values), 4),
                    round(np.max(valid_values), 4),
                    round(np.max(valid_values) - np.min(valid_values), 4),
                    tilt
                    
                ]
            except Exception as e:

                return [0, 0, 0, 0]  # Fallback bei Fehler


        def aggregate_energy(self, values):
            valid_values = values[values> -60]
            dealta_log_values = ProsodyEmbedding.Aggregation.delta_log(valid_values)

            return [np.mean(self.log_function(valid_values)),
                    np.max(valid_values),
                    np.min(valid_values),
                    np.std(valid_values),
                    np.mean(valid_values),

                    ]


        def aggregation(self, features):
            aggregated_features = []


            for feature_name, value in features.items():
                if feature_name in self.aggregation_function:
                    aggr_value = self.aggregation_function[feature_name](value)
                    aggregated_features.extend(aggr_value)
            
            return np.array(aggregated_features)




    def create_embedding(self, audio):
        if self.feature_segments is None:
            feature_segments = self.feature_segments

        features = {
            'f0': self.extract_f0(audio),
            'energy': self.extract_energy(audio)
    }

        segments = self.segment_audio(audio)

        all_features = []
        aggregator = self.Aggregation()



        for i, segment in enumerate (segments):
            features = {}

            for feature_name, segments_to_extract in self.feature_segments.items():
                if segments_to_extract == 'all':
                    features[feature_name] = getattr(self, f'extract_{feature_name}')(segment)
                elif segments_to_extract == 'first' and i < len(segments) -1:
                    features[feature_name] = getattr(self, f'extract_{feature_name}')(segment)
                elif segments_to_extract == 'last' and i == len(segments)-1:
                    features[feature_name] = getattr(self, f'extract_{feature_name}')(segment)
                elif isinstance(segments_to_extract, list) and i in segments_to_extract:
                    features[feature_name] = getattr(self, f'extract_{feature_name}')(segment)
            
            aggregated = aggregator.aggregation(features)
            all_features.append(aggregated)

            
        # Features aller Segmente kombinieren
        final_embedding = np.concatenate(all_features).flatten()

        # Ersetze nan-Werte mit 0 oder einem anderen sinnvollen Wert
        final_embedding = np.nan_to_num(final_embedding, nan=0.0)

        return final_embedding

