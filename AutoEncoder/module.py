import torch
from torch import nn as nn
from torch.nn import ModuleList


class FCNet(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        A list that specify the number of nodes for all hidden layers, should equal to n_layers
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = None,
        n_hidden: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()

        if n_layers is None:
            if n_hidden is None:
                n_hidden = [128]
                n_layers = 1
                print(
                    "n_hidden and n_layers not specified, use default"
                )
            else:
                n_layers = len(n_hidden)

        if n_layers != len(n_hidden):
            raise ValueError(
                "n_hidden not equal to n_layers"
            )

        if len(n_hidden) < 1:
            raise ValueError(
                "Should at least have 1 hidden layer"
            )
        # generate full layers_dim list
        layers_dim = [n_in] + n_hidden + [n_out]
        # iterate through layers
        fc_layers = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            fc_layers += [nn.Linear(in_features=n_in, out_features=n_out, bias=bias),
                          nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                          nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                          activation_fn() if use_activation else None,
                          nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None]
        fc_layers_removed_none = [x for x in fc_layers if x]

        self.fc_layers = nn.Sequential(*fc_layers_removed_none)

    def forward(self, x: torch.Tensor):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        x: torch.Tensor

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """

        return self.fc_layers(x)


# Encoder for RNA seq
class EncoderRNA(nn.Module):
    """
    Encodes RNA data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    **kwargs
        Keyword args for :class:`FCNet`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = None,
        n_hidden: list = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if n_layers is None:
            if n_hidden is None:
                n_hidden = [128]
                n_layers = 1
                print(
                    "n_hidden and n_layers not specified, use default"
                )
            else:
                n_layers = len(n_hidden)

        if n_layers != len(n_hidden):
            raise ValueError(
                "n_hidden not equal to n_layers"
            )

        if len(n_hidden) < 1:
            raise ValueError(
                "Should at least have 1 hidden layer"
            )
        self.encoder = FCNet(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network

        Parameters
        ----------
        x
            tensor with shape (n_input,)

        Returns
        -------
        q
            tensors of shape ``(n_latent,)``

        """
        # Parameters for latent distribution
        q = self.encoder(x)
        return q


# Decoder for RNA seq
class DecoderRNA(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = None,
        n_hidden: list = None,
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        if n_layers is None:
            if n_hidden is None:
                n_hidden = [128]
                n_layers = 1
                print(
                    "n_hidden and n_layers not specified, use default"
                )
            else:
                n_layers = len(n_hidden)

        if n_layers != len(n_hidden):
            raise ValueError(
                "n_hidden not equal to n_layers"
            )

        if len(n_hidden) < 1:
            raise ValueError(
                "Should at least have 1 hidden layer"
            )

        self.decoder = FCNet(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, z: torch.Tensor
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``

        Returns
        -------
        y
            tensors of shape ``(n_output,)``

        """
        # The decoder returns values for the parameters of the ZINB distribution
        y = self.decoder(z)
        return y


# Use different decoder for different dataset
class MultiDecoderRNA(nn.Module):
    """
        Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

        Uses a fully-connected neural network of ``n_hidden`` layers.

        Parameters
        ----------
        n_heads:
            The number of conditioned decoders
        n_input
            The dimensionality of the input (latent space)
        n_output
            The dimensionality of the output (data space)
        n_layers_conditioned
            The number of fully-connected hidden layers of conditioned decoders,
             should be a list, each element will be int
        n_hidden_conditioned
            The number of nodes for hidden layers of conditioned decoders,
             should be a list, each element will be a list
        n_layers_shared
            The number of fully-connected hidden layers of shared decoders,
             should be an int
        n_hidden_shared
            The number of nodes for hidden layers of shared decoders,
             should be a list, each element will be an int
        dropout_rate
            Dropout rate to apply to each of the hidden layers
        use_batch_norm
            Whether to use batch norm in layers
        use_layer_norm
            Whether to use layer norm in layers
        """
    def __init__(
        self,
        n_heads: int,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: list = None,
        n_hidden_shared: list = None,
        n_layers_conditioned: list = None,
        n_layers_shared: int = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        if n_layers_shared is None:
            if n_hidden_shared is None:
                n_hidden_shared = [128]
                n_layers_shared = 1
                print(
                    "n_hidden_shared and n_layers_shared not specified, use default"
                )
            else:
                n_layers_shared = len(n_hidden_shared)

        if n_layers_conditioned is None:
            if n_hidden_conditioned is None:
                n_hidden_conditioned = [[128]] * n_heads
                n_layers_conditioned = [1] * n_heads
                print(
                    "n_layers_conditioned and n_layers_conditioned not specified, use default"
                )
            else:
                n_layers_shared = [len(n_hidden) for n_hidden in n_hidden_conditioned]

        if n_layers_shared != len(n_hidden_shared):
            raise ValueError(
                "n_hidden_shared not equal to n_layers_shared"
            )

        for i in range(n_heads):
            if not n_layers_conditioned[i] == len(n_hidden_conditioned[i]):
                raise ValueError(
                    f"n_hidden_conditioned[{i}] not equal to n_layers_conditioned[{i}]"
                )
            if len(n_hidden_conditioned[i]) < 1:
                raise ValueError(
                    f"n_hidden_conditioned[{i}] should at least have 1 shared hidden layer"
                )

        self.decoders_conditioned = ModuleList(
            [
                FCNet(
                    n_in=n_input,
                    n_out=n_output,
                    n_layers=n_layers_conditioned[i],
                    n_hidden=n_hidden_conditioned[i],
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                )
                for i in range(n_heads)
            ]
        )

        self.decoder_shared = FCNet(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers_shared,
            n_hidden=n_hidden_shared,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self,
        z: torch.Tensor,  # should be batch x genes
        dataset_id: torch.Tensor,  # should be batch dim
    ):
        y_w = self.decoder_shared(z)
        y_v = torch.tensor([], device=y_w.device)
        for i in dataset_id:
            y_v = torch.cat((y_v, self.decoders_conditioned[i](z[i, :].view((1, -1)))))
        y = y_v + y_w

        return y_w, y_v, y

