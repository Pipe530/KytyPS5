#include "additionalContentDialog.h"

#include "configuration.h"
#include "gameContent.h"

#include <QDesktopServices>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QUrl>
#include <QVBoxLayout>

#include <map>

namespace {

struct DlcEntry {
	QString content_id;
	QString package_dir;
	QString title;
};

bool IsValidContentId(const QString& id) {
	static const QRegularExpression re(QStringLiteral("^[A-Z]{2}[0-9]{4}-[A-Z]{4}[0-9]{5}_[0-9]{2}-[A-Z0-9]{16}$"));
	return id.size() == 36 && re.match(id).hasMatch();
}

QString LabelOf(const QString& content_id) {
	const auto dash = content_id.lastIndexOf('-');
	return dash < 0 ? QString() : content_id.mid(dash + 1);
}

QString TitleOf(const QJsonObject& root) {
	const auto loc = root.value(QStringLiteral("localizedParameters")).toObject();
	QString    title = loc.value(loc.value(QStringLiteral("defaultLanguage")).toString())
	                    .toObject()
	                    .value(QStringLiteral("titleName"))
	                    .toString();
	for (auto it = loc.begin(); title.isEmpty() && it != loc.end(); ++it) {
		title = it.value().toObject().value(QStringLiteral("titleName")).toString();
	}
	return title.isEmpty() ? root.value(QStringLiteral("titleName")).toString() : title;
}

void AddEntry(std::map<QString, DlcEntry>& entries, const QString& content_id,
              const QString& package_dir, const QString& title) {
	const auto label = LabelOf(content_id);
	if (!IsValidContentId(content_id) || label.isEmpty()) {
		return;
	}
	auto it = entries.find(label);
	if (it == entries.end()) {
		entries[label] = {content_id, package_dir, title};
	} else if (it->second.package_dir.isEmpty() && !package_dir.isEmpty()) {
		it->second.package_dir = package_dir;
	}
}

void ParseParamJson(const QString& path, std::map<QString, DlcEntry>& entries) {
	QFile f(path);
	if (!f.open(QIODevice::ReadOnly)) {
		return;
	}
	const auto root = QJsonDocument::fromJson(f.readAll()).object();
	const auto id   = root.value(QStringLiteral("contentId")).toString();
	if (id.isEmpty()) {
		return;
	}
	QDir dir = QFileInfo(path).dir();
	if (dir.dirName() == QStringLiteral("sce_sys")) {
		dir.cdUp();
	}
	AddEntry(entries, id, dir.absolutePath(), TitleOf(root));
}

QLabel* IconLabel(const QPixmap& pixmap, int size, QWidget* parent) {
	auto* label = new QLabel(parent);
	label->setFixedSize(size, size);
	if (pixmap.isNull()) {
		label->setStyleSheet(QStringLiteral("background: #3a3d45;"));
	} else {
		label->setPixmap(pixmap.scaled(size, size, Qt::KeepAspectRatioByExpanding,
		                               Qt::SmoothTransformation));
	}
	return label;
}

const char* const STYLE =
    "QDialog { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1d1f24, stop:1 #101114); }"
    "QLabel { color: #ffffff; background: transparent; }"
    "QLabel#subtle { color: #9aa0aa; }"
    "QLabel#section { color: #c9ced6; font-size: 15px; }"
    "QListWidget { background: transparent; border: none; outline: none; }"
    "QListWidget::item { border-bottom: 1px solid #2c2f36; }"
    "QListWidget::item:hover { background: rgba(255,255,255,18); }"
    "QListWidget::item:selected { background: rgba(255,255,255,30); }"
    "QPushButton { color: #ffffff; background: #2c2f36; border: none; border-radius: 15px; "
    "padding: 7px 20px; }"
    "QPushButton:hover { background: #3d414b; }";

} // namespace

AdditionalContentDialog::AdditionalContentDialog(const Configuration& game, const QString& folder,
                                                 QWidget* parent)
    : QDialog(parent), m_title_id(game.title_id.trimmed().toUpper()), m_game_name(game.name),
      m_folder(folder.trimmed()) {
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle(tr("Additional Content - %1").arg(game.name));
	resize(820, 560);
	setStyleSheet(QString::fromLatin1(STYLE));

	const auto icon_data = GameContent::ReadFile(
	    game.basedir, QStringLiteral("sce_sys/icon0.png"), GameContent::MaxImageSize);
	if (!icon_data.isEmpty()) {
		m_game_icon.loadFromData(icon_data);
	}

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(36, 28, 36, 24);
	layout->setSpacing(10);

	auto* header = new QHBoxLayout;
	header->setSpacing(16);
	header->addWidget(IconLabel(m_game_icon, 72, this));
	auto* titles = new QVBoxLayout;
	titles->setSpacing(2);
	auto* name = new QLabel(m_game_name, this);
	name->setStyleSheet(QStringLiteral("font-size: 24px;"));
	auto* subtitle = new QLabel(m_title_id, this);
	subtitle->setObjectName(QStringLiteral("subtle"));
	subtitle->setStyleSheet(QStringLiteral("font-size: 15px;"));
	titles->addStretch();
	titles->addWidget(name);
	titles->addWidget(subtitle);
	titles->addStretch();
	header->addLayout(titles, 1);
	m_count = new QLabel(this);
	m_count->setObjectName(QStringLiteral("subtle"));
	m_count->setStyleSheet(QStringLiteral("font-size: 15px;"));
	header->addWidget(m_count, 0, Qt::AlignTop);
	layout->addLayout(header);

	layout->addSpacing(18);
	auto* section = new QLabel(tr("Add-ons"), this);
	section->setObjectName(QStringLiteral("section"));
	layout->addWidget(section);

	m_list = new QListWidget(this);
	m_list->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
	m_list->setFrameShape(QFrame::NoFrame);
	layout->addWidget(m_list, 1);

	m_empty = new QLabel(this);
	m_empty->setObjectName(QStringLiteral("subtle"));
	m_empty->setAlignment(Qt::AlignCenter);
	m_empty->setWordWrap(true);
	m_empty->setStyleSheet(QStringLiteral("font-size: 16px;"));
	layout->addWidget(m_empty, 1);

	auto* buttons = new QHBoxLayout;
	buttons->addStretch();
	auto* refresh = new QPushButton(tr("Refresh"), this);
	auto* open    = new QPushButton(tr("Open folder"), this);
	auto* close   = new QPushButton(tr("Close"), this);
	buttons->addWidget(refresh);
	buttons->addWidget(open);
	open->setVisible(!m_folder.isEmpty());
	buttons->addWidget(close);
	layout->addLayout(buttons);

	connect(refresh, &QPushButton::clicked, this, &AdditionalContentDialog::Scan);
	connect(open, &QPushButton::clicked, this, &AdditionalContentDialog::OpenFolder);
	connect(close, &QPushButton::clicked, this, &QDialog::close);

	Scan();
}

void AdditionalContentDialog::Scan() {
	m_list->clear();

	std::map<QString, DlcEntry> entries;
	if (!m_folder.isEmpty() && QDir(m_folder).exists()) {
		QDirIterator it(m_folder, QDir::Files, QDirIterator::Subdirectories);
		while (it.hasNext()) {
			const auto path = it.next();
			const auto name = QFileInfo(path).fileName();
			if (name.compare(QStringLiteral("param.json"), Qt::CaseInsensitive) == 0) {
				ParseParamJson(path, entries);
			}
		}
	}

	int count = 0;
	for (const auto& [label, entry]: entries) {
		if (!m_title_id.isEmpty() && !entry.content_id.contains(m_title_id, Qt::CaseInsensitive)) {
			continue;
		}

		QPixmap icon = m_game_icon;
		if (!entry.package_dir.isEmpty()) {
			QPixmap own(QDir(entry.package_dir).filePath(QStringLiteral("sce_sys/icon0.png")));
			if (!own.isNull()) {
				icon = own;
			}
		}

		auto* item = new QListWidgetItem(m_list);
		item->setSizeHint(QSize(0, 104));
		item->setToolTip(entry.content_id);

		auto* row = new QWidget(m_list);
		auto* h   = new QHBoxLayout(row);
		h->setContentsMargins(8, 8, 12, 8);
		h->setSpacing(18);
		h->addWidget(IconLabel(icon, 88, row));
		auto* title = new QLabel(entry.title.isEmpty() ? label : entry.title, row);
		title->setStyleSheet(QStringLiteral("font-size: 19px;"));
		h->addWidget(title, 1);
		auto* status =
		    new QLabel(entry.package_dir.isEmpty() ? tr("Unlocked") : tr("Installed"), row);
		status->setObjectName(QStringLiteral("subtle"));
		status->setStyleSheet(QStringLiteral("font-size: 16px;"));
		h->addWidget(status);
		m_list->setItemWidget(item, row);
		count++;
	}

	m_count->setText(tr("%n add-on(s)", nullptr, count));
	m_list->setVisible(count > 0);
	m_empty->setVisible(count == 0);
	if (m_folder.isEmpty()) {
		m_empty->setText(tr("No add-ons folder is set.\nOpen Global settings and set the "
		                    "Add-ons folder to the location of your add-on content."));
	} else if (QDir(m_folder).exists()) {
		m_empty->setText(tr("No add-ons found for this game in:\n%1")
		                     .arg(QDir::toNativeSeparators(m_folder)));
	} else {
		m_empty->setText(tr("The add-ons folder does not exist:\n%1\nCheck the path in Global "
		                    "settings.")
		                     .arg(QDir::toNativeSeparators(m_folder)));
	}
}

void AdditionalContentDialog::OpenFolder() const {
	if (m_folder.isEmpty() || !QDir(m_folder).exists()) {
		return;
	}
	QDesktopServices::openUrl(QUrl::fromLocalFile(m_folder));
}
