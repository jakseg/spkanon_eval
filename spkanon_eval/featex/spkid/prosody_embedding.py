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
        # Gesamtlänge berechnen
        total_length = len(audio)
        
        # In 3 gleiche Segmente aufteilen
        segment_length = total_length // 3
        
        segments = []
        for i in range(3):
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]
            segments.append(segment)

        # Hier Anzahl der Segmente printen
        print(f"Number of segments: {len(segments)}")
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
        #duration = self.extract_duration(segment)

        features = {
            'f0' : f0,
            'energy' : energy,
           # 'duration' : duration
        }

        return features



    def aggregation(self, features):

        aggregated_features = []
        
        # Für jedes Feature im Dictionary
        for feature_name, feature_values in features.items():
            # Nur berechnen wenn es ein Array oder Liste ist
            if isinstance(feature_values, (np.ndarray, list)):
                # Ignoriere Null-Werte für f0
                if feature_name == 'f0':
                    valid_values = feature_values[feature_values > 0]
                else:
                    valid_values = feature_values
                    
                # Berechne Statistiken
                feature_stats = [
                    np.mean(valid_values),
                    np.std(valid_values),
                    np.min(valid_values),
                    np.max(valid_values)
                ]

                # Füge Statistiken zur Liste hinzu
                aggregated_features.extend(feature_stats)
                
        return np.array(aggregated_features)



    def create_embedding(self, audio):
        segments = self.segment_audio(audio)

        all_features = []

        for i, segment in enumerate (segments):
            features = self.prosody_extraction(segment)
            aggregated = self.aggregation(features)
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
    embedding = prosody_embedding.create_embedding(audio)

    # Embedding ausgeben
    print(embedding)



