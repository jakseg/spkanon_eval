import parselmouth
import numpy as np



class ProsodyEmbedding:
    def __init__(self, config):
        """
        diverse Parameter:
        sample_rate
        """
        self.pitch_floor = config["pitch_floor"]
        self.pitch_ceiling = config["pitch_ceiling"]
        self.window_length = config["window_length"]
        self.time_step = config["time_step"]
        self.intensity_value = config["intensity_value"]
        self.config = config

    def segment_audio(self, audio):
        #Global Relative Time Intervals Approach
        # Gesamtlänge berechnen
        total_length = len(audio)
        segments = []

        # In 3 gleiche Segmente aufteilen
        segment_length = total_length // 3
        
        for i in range(3):
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]
            segments.append(segment)

        #Gesamte utterance als letztes Segment hinzufügen
        segments.append(audio[:])

        return segments


    
    def extract_f0(self, segment):
        sound = parselmouth.Sound(segment)
        pitch = sound.to_pitch(pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling, time_step=self.time_step)
        pitch_values = pitch.selected_array['frequency']
        return pitch_values


    def extract_energy(self, segment):
        sound = parselmouth.Sound(segment)
        intensity = sound.to_intensity(minimum_pitch=50, time_step=0.01)
        intensity_values = intensity.values[0]
        return intensity_values


    def prosody_extraction(self, segment):
        f0 = self.extract_f0(segment)
        energy = self.extract_energy(segment)


        features = {
            'f0' : f0,
            'energy' : energy,
           # 'duration' : duration
        }
        #print("Features dictionary:", features)
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

        def aggregate_f0(self, values):
            valid_values = values[values> 0]

            return [
                np.mean(valid_values),
                np.max(valid_values),
                np.max(valid_values) - np.min(valid_values)
            ]


        def aggregate_energy(self, values):
            valid_values = values[values> -60]

            return [np.mean(self.log_function(valid_values))]


        def aggregation(self, features):
            aggregated_features = []

            for feature_name, value in features.items():
                if feature_name in self.aggregation_function:
                    aggr_value = self.aggregation_function[feature_name](value)
                    aggregated_features.extend(aggr_value)
            #print(aggregated_features)
            return np.array(aggregated_features)




    def create_embedding(self, audio, feature_segments={'f0':'all', 'energy':'last'}):
        segments = self.segment_audio(audio)
        #print("Segments shape:", [len(seg) for seg in segments])
        all_features = []
        aggregator = self.Aggregation()

        #print("Segments shape:", [len(seg) for seg in segments])

        for i, segment in enumerate (segments):
            print(f"Segment {i} length:", len(segment))
            features = {}

            for feature_name, segments_to_extract in feature_segments.items():
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
        return final_embedding



if  __name__ == "__main__":
    path = "/path" 
    
    sound = parselmouth.Sound(path)
    
    config = {
        'pitch_floor': 75,
        'pitch_ceiling': 500,
        'window_length': 0.03,
        'time_step': 0.01,
        'intensity_value': 50
    }


    sound = parselmouth.Sound(path)
    audio = sound.values[0] # Audiodaten als Array extrahieren

    # Prosody Embedding Objekt erstellen
    prosody_embedding = ProsodyEmbedding(config)

    # Embedding erstellen
    embedding = prosody_embedding.create_embedding(audio, feature_segments={'f0': 'all', 'energy': 'all'})

    # Embedding ausgeben
    print(embedding)



