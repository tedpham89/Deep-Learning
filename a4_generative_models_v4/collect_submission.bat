@echo off
if exist assignment_4_submission.zip del /F /Q assignment_4_submission.zip

tar -a -c -f assignment_4_submission.zip ^
    configs ^
    config.py ^
    models/*.py ^
    losses/*.py ^
    utils/*.py ^
    outputs/vae/vae_mnist.pth ^
    outputs/vae/vae_fashionmnist.pth ^
    outputs/gan/discriminator_mnist.pth ^
    outputs/gan/generator_mnist.pth ^
    outputs/gan/discriminator_fashionmnist.pth ^
    outputs/gan/generator_fashionmnist.pth ^
    outputs/diffusion/diffusion_net_mnist.pth ^
    outputs/diffusion/diffusion_net_fashionmnist.pth ^
    train.py ^
    trainer_vae.py ^
    trainer_gan.py ^
    simple_diffusion.py ^
    trainer_diffusion.py
