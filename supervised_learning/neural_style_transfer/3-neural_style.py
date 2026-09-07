    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost

        Sets instance attributes:
            gram_style_features: list of gram matrices from style layers
            content_feature: content layer output of content image
        """
        # Get all outputs from style image
        style_outputs = self.model(self.style_image)

        # Calculate gram matrices for style layers (first 5 outputs)
        self.gram_style_features = []
        for i in range(len(self.style_layers)):
            gram = self.gram_matrix(style_outputs[i])
            self.gram_style_features.append(gram)

        # Get all outputs from content image
        content_outputs = self.model(self.content_image)

        # Extract content feature (last output)
        self.content_feature = content_outputs[-1]
