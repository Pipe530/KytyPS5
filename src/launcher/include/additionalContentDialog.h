#ifndef ADDITIONAL_CONTENT_DIALOG_H
#define ADDITIONAL_CONTENT_DIALOG_H

#include <QDialog>
#include <QPixmap>
#include <QString>

class Configuration;
class QLabel;
class QListWidget;

class AdditionalContentDialog final: public QDialog {
public:
	AdditionalContentDialog(const Configuration& game, const QString& folder, QWidget* parent = nullptr);
	~AdditionalContentDialog() override = default;

private:
	void Scan();
	void OpenFolder() const;

	QString      m_title_id;
	QString      m_game_name;
	QString      m_folder;
	QPixmap      m_game_icon;
	QLabel*      m_count  = nullptr;
	QLabel*      m_empty  = nullptr;
	QListWidget* m_list   = nullptr;
};

#endif // ADDITIONAL_CONTENT_DIALOG_H
