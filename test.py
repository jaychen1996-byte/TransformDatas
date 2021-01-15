import pygame

# from pygame import image

if __name__ == '__main__':
    pygame.init()
    # images = pygame.image.load("s2_label_gt/test_del.png")
    images = pygame.image.load("test_pad.png")
    # images = pygame.image.load("label_gt_pic.png")
    images.set_colorkey((0, 0, 0))
    # images = pygame.Surface.convert_alpha(images)
    bg = pygame.image.load("s3_bg_gt/2020_09_16_13_58_02_620.bmp")
    rect1 = pygame.Rect(0, 0, 100, 100)
    bg.blit(images, (100, 100), rect1)
    pygame.image.save(bg, "test.png")
