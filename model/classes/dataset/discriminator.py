from pylab import rcParams
# set matplotlib rcParams to set fig size
rcParams['figure.figsize'] = 15, 15


def generate_decoder_values(pic_1, pic_2, size, encoder, decoder):
    """Interpolates between pic_1 and pic_2 by interpolating between their embeddings from the autoencoder.
    pic_1(np.array) - from sample_points
    pic_2(same as above)
    size(integer) - number of intermediate images
    """
    emb_1, emb_2 = [encoder.predict(np.expand_dims(pic, axis=0)) for pic in [pic_1, pic_2]]
    decoder_values = []
    # Sample random points between [0, 1], then sort them
    # Effectively taking random jumps from pic_1 to pic_2 as we interpolate between them
    # Looks decent though.
    # C = np.sort(np.random.rand(size))
    
    # Take uniform steps
    C = np.linspace(0, 1, num=size)
    for i in range(C.shape[0]):
        d_val = decoder.predict(C[i] * (emb_2 - emb_1) + emb_1)[0]
        decoder_values.append(d_val)
    print(decoder_values[0].shape)
    return decoder_values, C

def gif_range(pic_1, pic_2, num_imgs, encoder, decoder, fname=None, save=True):
    """
    Makes a GIF that interpolates between pic_1 and pic_2 by sampling from the line between their embeddings.
    
    imgs(list of np.arrays): the intermediary sampled pics
    C(np.array): values sampled to interpolate
    pic_1/pic_2: np.array of actual image
    """
    imgs, C = generate_decoder_values(pic_1, pic_2, num_imgs, encoder, decoder)
    fontdict = {'fontsize':25, 'fontweight': 5}
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(pic_1)
    ax[0].set_title("A", fontdict=fontdict)
    ax[2].imshow(pic_2)
    ax[2].set_title("B", fontdict=fontdict)

    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))

    def update(i):
        """This is what creates the different frames."""
        label = '{0:2.0f}% A, {1:2.0f}% B'.format(100 - 100 * C[i], 100 * C[i])
        # Useful for debugging
        print(label)
        
        # Intermediary Encoding
        ax[1].imshow(imgs[i])
        ax[1].set_title(label, fontdict)
        
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        return ax
    
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_imgs), interval=70)
    if save:
        if not fname:
            fname = '{}-decoder_predictions-new.gif'.format(BACKEND)
        anim.save(fname, dpi=80, writer='imagemagick')
        print("Saved to {}".format(fname))
    plt.show()
    plt.close(fig)
