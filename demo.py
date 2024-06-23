import gradio as gr
import lightning as L
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.io import read_video
from torchvision.transforms.v2 import Resize, Normalize, UniformTemporalSubsample
from models import VideoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

classes = ['America',
 'Amarelo',
 'Espelho',
 'Ruim',
 'Barulho',
 'Vacina',
 'Vontade',
 'Conhecer',
 'Sapo',
 'Acontecer',
 'Bala',
 'Filho',
 'Medo',
 'Aluno',
 'Esquina',
 'Banco',
 'Cinco',
 'Banheiro',
 'Aproveitar',
 'Maca']
classes.sort()

def preprocess(video_as_mp4):
    frames = read_video(video_as_mp4)[0].float()
    frames = frames.permute(3, 0, 1, 2)
    frames = frames.permute(1, 0, 2, 3)
    frames = Resize((224, 224))(frames)
    frames = Normalize((118.4939, 118.4997, 118.5007), (57.2457, 57.2454, 57.2461))(
        frames
    )
    frames = UniformTemporalSubsample(16)(frames)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)
    return frames


def load_model():
    model = VideoModel.load_from_checkpoint(
        "/mnt/E-SSD/BRACIS-2024/lightning_logs/timesformer_sample_from_32/version_0/checkpoints/best_model.ckpt"
    )
    model.eval()
    model = model.to(device)
    return model


def video_identity(video):
    print(video)
    video = preprocess(video)
    print(video.shape)

    with torch.no_grad():
        video = video
        model = load_model()
        output = model(video)
        print(output)
    output = torch.nn.functional.softmax(output.logits, dim=1)
    print(output)
    topk_prob, topk_label = torch.topk(output, 5)
    print(topk_prob, topk_label)
    probs = topk_prob.cpu().numpy().flatten()
    labels = topk_label.cpu().numpy().flatten()
    return {
        classes[label]: float(prob) for label, prob in zip(labels, probs)
    }


demo = gr.Interface(
    video_identity,
    gr.Video(),
    gr.Label(
        num_top_classes=5,
    ),
)

if __name__ == "__main__":
    model = load_model()

    demo.launch(inline=True, share=True)
