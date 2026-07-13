from proj.func import Configs
from dataclasses import asdict
from train import trainer as trainer_


def main(configs: Configs):

    model = configs.model(**asdict(configs.model_config)).to(configs.train_config.config['device'])

    print('Model Uploaded')
    print(model)

    trainer = trainer_.Trainer(**configs.train_config.config)
    trainer.set_model(model, 'test')

    trainer_data = trainer_.TrainerData(**configs.train_config.data)
    trainer_data.prepare_loaders()

    predictor = trainer_.Predictor()

    trainer.train(trainer_data.train_loader,
                  trainer_data.val_loader,
                  param_path_dir=trainer_data.save_param_dir)

    y_test, y_hat = predictor.predict(model, 
                      trainer_data.test_loader,
                      trainer.best_model_val_loss_param)

    predictor.calc_perf_stats(y_test, y_hat)


    