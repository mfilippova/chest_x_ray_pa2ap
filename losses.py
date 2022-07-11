import torch


def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Adversarial loss for the Discriminator
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X
        adv_criterion: the adversarial loss function
    '''
    disc_fake_X_hat = disc_X(fake_X.detach())
    disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
    disc_real_X_hat = disc_X(real_X)
    disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2    
    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Adversarial loss for the Generator
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y
        gen_XY: the generator for class X to Y
        adv_criterion: the adversarial loss function
    '''
    fake_Y = gen_XY(real_X)
    disc_fake_Y_hat = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))    
    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Identity loss for the Generator
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X
        identity_criterion: the identity loss function
    '''
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)
    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Cycle consistency loss for the Generator
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X
        cycle_criterion: the cycle consistency loss function
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    return cycle_loss, cycle_X


def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10, use_identity=False):
    '''
    Total loss for the Generator
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B
        gen_BA: the generator for class B to A
        disc_A: the discriminator for class A
        disc_B: the discriminator for class B
        adv_criterion: the adversarial loss function
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; returns pixel difference of two images
        cycle_criterion: the cycle consistency loss function
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''
    adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    gen_adversarial_loss = adv_loss_AB + adv_loss_BA

    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    gen_cycle_loss = cycle_loss_AB + cycle_loss_BA

    gen_loss = lambda_cycle * gen_cycle_loss + gen_adversarial_loss
    
    if use_identity:
        identity_loss_A, identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
        identity_loss_B, identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)
        gen_identity_loss = identity_loss_A + identity_loss_B
        
        gen_loss += lambda_identity * gen_identity_loss

    return gen_loss, fake_A, fake_B
